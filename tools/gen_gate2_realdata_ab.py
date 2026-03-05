#!/usr/bin/env python3
"""Generate Gate2RunInputV1 JSON inputs from local real-data text (E0 vs E1)."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from e8_roots import build_e8_roots, write_verification_record
from realdata_label_utils import count_labels_records, normalize_label

UNITIZATION_V1 = "sentence_split_v1"
UNITIZATION_V2_MIN4 = "sentence_split_v2_min4"
SUPPORTED_UNITIZATION_IDS = (UNITIZATION_V1, UNITIZATION_V2_MIN4)
EMBEDDING_BACKEND_ID = "minilm_all_MiniLM_L6_v2_384d_v1"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
E0_POSTPROC_ID = "chunk_sequential_48x8_v1"
E1_POSTPROC_ID = "gaussian_proj_d8_seed7_v1"
E2_POSTPROC_ID = "e8_softsnap_chunk48_k3_beta12_v1"
E0_AGGREGATE_ID = "mean_then_chunk"
E2_AGGREGATE_ID = "mean_then_chunk"
E2_K = 3
E2_BETA = 12.0
U64_MAX = (1 << 64) - 1
NORM_EPS = 1e-12
PRIMARY_SPLIT_RE = re.compile("[.!?\\u3002\\uFF01\\uFF1F\\n]+")
SECONDARY_SPLIT_RE = re.compile("[,;\\uFF0C\\uFF1B]+")
SECONDARY_V2_SPLIT_RE = re.compile("[,;:\\uFF0C\\uFF1B\\uFF1A]+")
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate real-data Gate2RunInputV1 JSON for E0/E1 vec8 postproc A/B."
    )
    parser.add_argument(
        "--input-jsonl",
        default="data/realdata/phase3_q2_realdata_mini.jsonl",
        help="Input JSONL path.",
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-units", type=int, default=24)
    parser.add_argument("--unitization-id", default=UNITIZATION_V1)
    parser.add_argument("--balanced", type=int, choices=(0, 1), default=1)
    parser.add_argument("--n0", type=int, default=100)
    parser.add_argument("--n1", type=int, default=100)
    parser.add_argument("--out-dir", default="data/realdata")
    parser.add_argument("--run-id-e0", default="realdata_E0_chunk")
    parser.add_argument("--run-id-e1", default="realdata_E1_proj")
    parser.add_argument("--run-id-e2", default="realdata_E2_e8softsnap")
    args = parser.parse_args()

    if args.n_samples <= 0:
        parser.error("--n-samples must be > 0")
    if args.max_units <= 0:
        parser.error("--max-units must be > 0")
    if args.n0 < 0 or args.n1 < 0:
        parser.error("--n0/--n1 must be >= 0")
    if args.unitization_id not in SUPPORTED_UNITIZATION_IDS:
        parser.error(
            "--unitization-id must be one of: "
            + ", ".join(SUPPORTED_UNITIZATION_IDS)
        )
    if args.unitization_id == UNITIZATION_V2_MIN4 and args.max_units < 4:
        parser.error("--max-units must be >= 4 for sentence_split_v2_min4")
    return args


def is_u64_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= U64_MAX


def parse_label(value: Any) -> Optional[int]:
    return normalize_label(value)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")


def write_anchor_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "label", "anchor_mean", "anchor_p90", "anchor_max"]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": int(row["sample_id"]),
                    "label": "" if row["label"] is None else int(row["label"]),
                    "anchor_mean": f"{float(row['anchor_mean']):.17e}",
                    "anchor_p90": f"{float(row['anchor_p90']):.17e}",
                    "anchor_max": f"{float(row['anchor_max']):.17e}",
                }
            )


def build_fallback_rows() -> List[Dict[str, Any]]:
    return [
        {
            "sample_id": 1001,
            "answer": (
                "The report states the release passed smoke tests. "
                "However, two warnings remain in telemetry. "
                "We should track them in the next sprint. "
                "The follow-up owner is still open."
            ),
            "label": 1,
        },
        {
            "sample_id": 1002,
            "answer": (
                "I reviewed the claim. "
                "The cited number is not present in the source document. "
                "The answer is likely unsupported. "
                "A correction should be issued."
            ),
            "label": 0,
        },
        {
            "sample_id": 1003,
            "answer": (
                "Migration is mostly complete, but one service still writes legacy fields. "
                "Until that service is updated, metrics can drift. "
                "Backfill checks should stay enabled."
            ),
            "label": None,
        },
        {
            "sample_id": 1004,
            "answer": (
                "The policy text is explicit. "
                "It allows retries after cooldown. "
                "It does not allow bypassing rate limits. "
                "The exception list remains empty."
            ),
            "label": 1,
        },
        {
            "sample_id": 1005,
            "answer": (
                "User feedback mentions slow search. "
                "Profiling shows index warmup dominates startup latency. "
                "Caching should reduce the first query delay. "
                "A benchmark run is planned."
            ),
            "label": None,
        },
        {
            "sample_id": 1006,
            "answer": (
                "This response invents a source and date. "
                "No citation was provided. "
                "The statement should be marked incorrect. "
                "Escalation is recommended."
            ),
            "label": 0,
        },
        {
            "sample_id": 1007,
            "answer": (
                "Deployment notes confirm the schema change landed yesterday. "
                "Read replicas were lagging briefly. "
                "All regions are now healthy. "
                "No rollback is required."
            ),
            "label": 1,
        },
        {
            "sample_id": 1008,
            "answer": (
                "The summary mixes two experiments. "
                "One used synthetic inputs, the other used real data. "
                "Conclusions should separate them. "
                "The chart legend needs an update."
            ),
            "label": None,
        },
        {
            "sample_id": 1009,
            "answer": (
                "The answer contradicts the quoted paragraph. "
                "It says the feature is enabled by default. "
                "The paragraph says default is disabled. "
                "The claim should be rejected."
            ),
            "label": 0,
        },
        {
            "sample_id": 1010,
            "answer": (
                "Unit tests cover expected paths. "
                "Edge-case handling still needs integration tests. "
                "Risk remains moderate. "
                "Release notes should reflect this."
            ),
            "label": 1,
        },
    ]


def maybe_create_fallback_dataset(path: Path) -> bool:
    if path.exists():
        return False
    rows = build_fallback_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            sample_id = row.get("sample_id")
            answer = row.get("answer")
            if not is_u64_int(sample_id):
                continue
            if not isinstance(answer, str) or not answer.strip():
                continue
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id {sample_id} in {path}")
            seen_ids.add(sample_id)
            records.append(
                {
                    "sample_id": int(sample_id),
                    "answer": answer,
                    "label": parse_label(row.get("label")),
                    "_file_order": len(records),
                }
            )
    return records


def split_units_primary(answer: str) -> List[str]:
    return [segment.strip() for segment in PRIMARY_SPLIT_RE.split(answer) if segment.strip()]


def split_units_secondary(answer: str, pattern: re.Pattern = SECONDARY_SPLIT_RE) -> List[str]:
    return [segment.strip() for segment in pattern.split(answer) if segment.strip()]


def split_units_char_chunks(answer: str, max_units: int, target_units: int = 3) -> List[str]:
    collapsed = SPACE_RE.sub(" ", answer).strip()
    if not collapsed:
        return []
    desired = max(1, min(max_units, target_units))
    # Deterministic fixed boundaries: floor(i*L/n) .. floor((i+1)*L/n)
    total_len = len(collapsed)
    desired = min(desired, total_len)
    chunks = []
    for i in range(desired):
        start = int((i * total_len) / desired)
        end = int(((i + 1) * total_len) / desired)
        chunk = collapsed[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks[:max_units]


def unitize_v1(answer: str, max_units: int) -> List[str]:
    units = split_units_primary(answer)
    if len(units) < 3:
        secondary = split_units_secondary(answer)
        if len(secondary) >= len(units):
            units = secondary
    if len(units) < 3:
        chunks = split_units_char_chunks(answer, max_units=max_units, target_units=3)
        if len(chunks) >= len(units):
            units = chunks
    if len(units) < 3 and units:
        # Gate2/Gate3 are more informative with at least 3 steps.
        while len(units) < 3:
            units.append(units[-1])
    return units[:max_units]


def unitize_v2_min4(answer: str, max_units: int) -> List[str]:
    units = split_units_primary(answer)
    if len(units) < 4:
        secondary = split_units_secondary(answer, pattern=SECONDARY_V2_SPLIT_RE)
        if len(secondary) >= len(units):
            units = secondary
    if len(units) < 4:
        chunks = split_units_char_chunks(answer, max_units=max_units, target_units=4)
        if len(chunks) >= len(units):
            units = chunks
    if len(units) < 4 and units:
        while len(units) < 4:
            units.append(units[-1])
    return units[:max_units]


def unitize(answer: str, max_units: int, unitization_id: str) -> List[str]:
    if unitization_id == UNITIZATION_V1:
        return unitize_v1(answer, max_units=max_units)
    if unitization_id == UNITIZATION_V2_MIN4:
        return unitize_v2_min4(answer, max_units=max_units)
    raise ValueError(f"unsupported unitization_id: {unitization_id}")


def normalize_vector(values: Sequence[float], eps: float = NORM_EPS) -> List[float]:
    total = 0.0
    for value in values:
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError("non-finite value before normalization")
        total += value_f * value_f
    norm = math.sqrt(total)
    if not math.isfinite(norm) or norm <= eps:
        raise ValueError("near-zero or non-finite norm in normalization")
    return [float(value) / norm for value in values]


def normalize8(values: Sequence[float]) -> List[float]:
    if len(values) != 8:
        raise ValueError("normalize8 expects length 8")
    return normalize_vector(values, eps=NORM_EPS)


def to_embedding_rows(encoded: Any, expected_rows: int) -> List[List[float]]:
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if expected_rows == 1 and isinstance(encoded, list):
        if encoded and isinstance(encoded[0], (int, float)):
            encoded = [encoded]
    if not isinstance(encoded, list):
        raise ValueError("model.encode returned an unsupported type")
    if len(encoded) != expected_rows:
        raise ValueError(
            f"model.encode returned {len(encoded)} rows, expected {expected_rows}"
        )
    out: List[List[float]] = []
    for row_idx, row in enumerate(encoded):
        if not isinstance(row, list):
            raise ValueError(f"embedding row {row_idx} has invalid type")
        if len(row) != 384:
            raise ValueError(f"embedding row {row_idx} has dim={len(row)}, expected 384")
        converted = [float(value) for value in row]
        for value in converted:
            if not math.isfinite(value):
                raise ValueError(f"embedding row {row_idx} contains non-finite value")
        out.append(converted)
    return out


def mean_embedding(rows: Sequence[Sequence[float]]) -> List[float]:
    if not rows:
        raise ValueError("cannot average empty embedding rows")
    dim = len(rows[0])
    acc = [0.0] * dim
    for row in rows:
        if len(row) != dim:
            raise ValueError("embedding dimension mismatch")
        for idx, value in enumerate(row):
            acc[idx] += float(value)
    n = float(len(rows))
    return [value / n for value in acc]


def e0_chunk_from_embeddings(emb_rows: Sequence[Sequence[float]]) -> List[List[float]]:
    avg = mean_embedding(emb_rows)
    if len(avg) != 384:
        raise ValueError("E0 expected 384D average embedding")
    ans_vec8: List[List[float]] = []
    for offset in range(0, 384, 8):
        ans_vec8.append(normalize8(avg[offset : offset + 8]))
    if len(ans_vec8) != 48:
        raise ValueError(f"E0 produced {len(ans_vec8)} rows; expected 48")
    return ans_vec8


def compute_p90(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute p90 for empty sequence")
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = int(math.ceil(0.9 * n)) - 1
    rank = max(0, min(rank, n - 1))
    return sorted_vals[rank]


def e2_soft_snap_chunk48(
    emb_rows: Sequence[Sequence[float]],
    roots: Sequence[Sequence[float]],
    k: int = E2_K,
    beta: float = E2_BETA,
) -> Tuple[List[List[float]], List[float]]:
    if k <= 0:
        raise ValueError("k must be > 0 for soft snap")
    avg = mean_embedding(emb_rows)
    if len(avg) != 384:
        raise ValueError("E2 expected 384D average embedding")

    steps: List[List[float]] = []
    anchors: List[float] = []

    for offset in range(0, 384, 8):
        block = normalize8(avg[offset : offset + 8])
        scores: List[Tuple[float, int]] = []
        for idx, root in enumerate(roots):
            score = 0.0
            for a, b in zip(block, root):
                score += float(a) * float(b)
            scores.append((score, idx))

        scores.sort(key=lambda item: (-item[0], item[1]))
        top = scores[: min(k, len(scores))]
        max_dot = top[0][0]

        exps = [math.exp(beta * (dot - max_dot)) for dot, _ in top]
        denom = sum(exps)
        if denom <= 0.0 or not math.isfinite(denom):
            raise ValueError("invalid softmax denominator in E2 soft snap")

        blended = [0.0] * 8
        for weight_raw, (dot, root_idx) in zip(exps, top):
            _ = dot
            weight = weight_raw / denom
            root = roots[root_idx]
            for i in range(8):
                blended[i] += weight * float(root[i])
        steps.append(normalize8(blended))

        d = max(-1.0, min(1.0, float(max_dot)))
        wedge = math.sqrt(max(0.0, 1.0 - d * d))
        theta = math.atan2(wedge, d)
        anchor = theta / math.pi
        anchors.append(anchor)

    if len(steps) != 48:
        raise ValueError(f"E2 produced {len(steps)} rows; expected 48")
    return steps, anchors


class SplitMix64:
    """Deterministic splitmix64 PRNG."""

    _MASK = 0xFFFFFFFFFFFFFFFF

    def __init__(self, seed: int) -> None:
        self.state = seed & self._MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        z = self.state
        z = (z ^ (z >> 30)) & self._MASK
        z = (z * 0xBF58476D1CE4E5B9) & self._MASK
        z = (z ^ (z >> 27)) & self._MASK
        z = (z * 0x94D049BB133111EB) & self._MASK
        z = (z ^ (z >> 31)) & self._MASK
        return z

    def next_f64_open01(self) -> float:
        # 53 random bits -> [0, 1). Avoid exact zero for Box-Muller log.
        raw = self.next_u64() >> 11
        value = raw * (1.0 / float(1 << 53))
        if value <= 0.0:
            return 1.0 / float(1 << 53)
        return value


def box_muller_pair(rng: SplitMix64) -> Tuple[float, float]:
    u1 = rng.next_f64_open01()
    u2 = rng.next_f64_open01()
    radius = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return radius * math.cos(theta), radius * math.sin(theta)


def build_projection_matrix(seed: int) -> List[List[float]]:
    rng = SplitMix64(seed)
    total = 8 * 384
    values: List[float] = []
    while len(values) < total:
        z0, z1 = box_muller_pair(rng)
        values.append(z0)
        if len(values) < total:
            values.append(z1)

    matrix: List[List[float]] = []
    idx = 0
    for _ in range(8):
        row = values[idx : idx + 384]
        idx += 384
        matrix.append(normalize_vector(row))
    return matrix


def matrix_hash_sha256(matrix: Sequence[Sequence[float]]) -> str:
    hasher = hashlib.sha256()
    for row in matrix:
        for value in row:
            hasher.update(struct.pack("<d", float(value)))
    return hasher.hexdigest()


def project_embedding(matrix: Sequence[Sequence[float]], emb: Sequence[float]) -> List[float]:
    if len(emb) != 384:
        raise ValueError("projection expects 384D embedding")
    out = []
    for row in matrix:
        if len(row) != 384:
            raise ValueError("projection matrix row must be 384D")
        acc = 0.0
        for weight, value in zip(row, emb):
            acc += float(weight) * float(value)
        out.append(acc)
    return normalize8(out)


def validate_gate2_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload.get("run_id"), str):
        raise ValueError("run_id must be string")
    unrelated = payload.get("explicitly_unrelated_sample_ids")
    if not isinstance(unrelated, list):
        raise ValueError("explicitly_unrelated_sample_ids must be list")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("samples must be list")
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("sample rows must be objects")
        sample_id = sample.get("sample_id")
        if not is_u64_int(sample_id):
            raise ValueError(f"sample_id must be u64 int, got {sample_id!r}")
        label = sample.get("sample_label")
        if label not in (0, 1, None):
            raise ValueError(f"sample_label must be 0/1/null for sample_id={sample_id}")
        if "answer_length" in sample:
            answer_length = sample["answer_length"]
            if not (isinstance(answer_length, int) and answer_length >= 0):
                raise ValueError(f"invalid answer_length for sample_id={sample_id}")
        ans_vec8 = sample.get("ans_vec8")
        if not isinstance(ans_vec8, list):
            raise ValueError(f"ans_vec8 must be list for sample_id={sample_id}")
        for row_idx, row in enumerate(ans_vec8):
            if not isinstance(row, list) or len(row) != 8:
                raise ValueError(
                    f"sample_id={sample_id} row={row_idx} must be an array of 8 floats"
                )
            for col_idx, value in enumerate(row):
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(
                        f"non-finite ans_vec8 value sample_id={sample_id} "
                        f"row={row_idx} col={col_idx}"
                    )


def record_sort_key(row: Dict[str, Any]) -> Tuple[int, int]:
    sample_id = row.get("sample_id")
    if is_u64_int(sample_id):
        return (0, int(sample_id))
    return (1, int(row.get("_file_order", 0)))


def select_records_unbalanced(records: List[Dict[str, Any]], n_samples: int) -> List[Dict[str, Any]]:
    sorted_records = sorted(records, key=record_sort_key)
    return sorted_records[:n_samples]


def select_records_balanced(
    records: List[Dict[str, Any]], n0: int, n1: int
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    label0 = sorted((row for row in records if row["label"] == 0), key=record_sort_key)
    label1 = sorted((row for row in records if row["label"] == 1), key=record_sort_key)
    chosen0 = label0[:n0]
    chosen1 = label1[:n1]

    selected = chosen0 + chosen1
    selected_sorted = sorted(selected, key=record_sort_key)

    meta = {
        "n0_requested": n0,
        "n1_requested": n1,
        "n0_available": len(label0),
        "n1_available": len(label1),
        "n0_selected": len(chosen0),
        "n1_selected": len(chosen1),
    }
    return selected_sorted, meta


def main() -> int:
    args = parse_args()

    input_jsonl = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fallback_created = maybe_create_fallback_dataset(input_jsonl)
    records = load_records(input_jsonl)
    if args.balanced == 1:
        selected, selection_meta = select_records_balanced(records, n0=args.n0, n1=args.n1)
        n_samples_requested = args.n0 + args.n1
    else:
        selected = select_records_unbalanced(records, args.n_samples)
        selection_meta = {
            "n0_requested": 0,
            "n1_requested": 0,
            "n0_available": len([row for row in records if row["label"] == 0]),
            "n1_available": len([row for row in records if row["label"] == 1]),
            "n0_selected": len([row for row in selected if row["label"] == 0]),
            "n1_selected": len([row for row in selected if row["label"] == 1]),
        }
        n_samples_requested = args.n_samples
    if not selected:
        raise ValueError("no usable samples found with non-empty answer text")
    selected_label_counts = count_labels_records(selected)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required. Install with: "
            "python -m pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    projection = build_projection_matrix(args.seed)
    projection_hash = matrix_hash_sha256(projection)
    e8_roots = build_e8_roots()
    e8_verify_path = (
        Path("attestations")
        / "realdata_ab"
        / f"{dt.date.today().isoformat()}_e8_roots_verify.txt"
    )
    verify_written_path, e8_verify_info, e8_verify_sha256 = write_verification_record(
        e8_roots, e8_verify_path
    )

    samples_e0: List[Dict[str, Any]] = []
    samples_e1: List[Dict[str, Any]] = []
    samples_e2: List[Dict[str, Any]] = []
    unit_counts: List[int] = []
    anchor_rows: List[Dict[str, Any]] = []
    anchor_mean_values: List[float] = []
    anchor_p90_values: List[float] = []
    anchor_max_values: List[float] = []

    for record in selected:
        sample_id = record["sample_id"]
        answer = record["answer"]
        label = record["label"]

        units = unitize(answer, max_units=args.max_units, unitization_id=args.unitization_id)
        if not units:
            continue
        unit_count = len(units)
        unit_counts.append(unit_count)

        encoded = model.encode(
            units,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        emb_rows = to_embedding_rows(encoded, expected_rows=unit_count)

        e0_rows = e0_chunk_from_embeddings(emb_rows)
        e1_rows = [project_embedding(projection, emb) for emb in emb_rows]
        e2_rows, e2_anchor_values = e2_soft_snap_chunk48(
            emb_rows=emb_rows, roots=e8_roots, k=E2_K, beta=E2_BETA
        )

        min_required_rows = 4 if args.unitization_id == UNITIZATION_V2_MIN4 else 3
        if len(e1_rows) < min_required_rows:
            raise ValueError(
                f"sample_id={sample_id} produced <{min_required_rows} E1 rows "
                "after fallback unitization"
            )
        if len(e2_rows) != 48:
            raise ValueError(f"sample_id={sample_id} produced invalid E2 rows")

        anchor_mean = sum(e2_anchor_values) / float(len(e2_anchor_values))
        anchor_p90 = compute_p90(e2_anchor_values)
        anchor_max = max(e2_anchor_values)
        anchor_rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "anchor_mean": anchor_mean,
                "anchor_p90": anchor_p90,
                "anchor_max": anchor_max,
            }
        )
        anchor_mean_values.append(anchor_mean)
        anchor_p90_values.append(anchor_p90)
        anchor_max_values.append(anchor_max)

        samples_e0.append(
            {
                "sample_id": sample_id,
                "ans_vec8": e0_rows,
                "sample_label": label,
                "answer_length": unit_count,
            }
        )
        samples_e1.append(
            {
                "sample_id": sample_id,
                "ans_vec8": e1_rows,
                "sample_label": label,
                "answer_length": unit_count,
            }
        )
        samples_e2.append(
            {
                "sample_id": sample_id,
                "ans_vec8": e2_rows,
                "sample_label": label,
                "answer_length": unit_count,
            }
        )

    if not samples_e0 or not samples_e1 or not samples_e2:
        raise ValueError("no samples survived preprocessing")

    payload_e0 = {
        "run_id": args.run_id_e0,
        "explicitly_unrelated_sample_ids": [],
        "samples": samples_e0,
    }
    payload_e1 = {
        "run_id": args.run_id_e1,
        "explicitly_unrelated_sample_ids": [],
        "samples": samples_e1,
    }
    payload_e2 = {
        "run_id": args.run_id_e2,
        "explicitly_unrelated_sample_ids": [],
        "samples": samples_e2,
    }
    validate_gate2_payload(payload_e0)
    validate_gate2_payload(payload_e1)
    validate_gate2_payload(payload_e2)

    out_e0 = out_dir / "out_E0_chunk.json"
    out_e1 = out_dir / "out_E1_proj.json"
    out_e2 = out_dir / "out_E2_snap.json"
    out_anchor_csv = out_dir / "e2_anchor_stats.csv"
    out_meta = out_dir / "ab_meta.json"

    write_json(out_e0, payload_e0)
    write_json(out_e1, payload_e1)
    write_json(out_e2, payload_e2)
    write_anchor_csv(out_anchor_csv, anchor_rows)

    unit_count_min = min(unit_counts)
    unit_count_max = max(unit_counts)
    unit_count_mean = sum(unit_counts) / float(len(unit_counts))

    anchor_summary = {
        "anchor_mean": {
            "min": min(anchor_mean_values),
            "max": max(anchor_mean_values),
            "mean": sum(anchor_mean_values) / float(len(anchor_mean_values)),
        },
        "anchor_p90": {
            "min": min(anchor_p90_values),
            "max": max(anchor_p90_values),
            "mean": sum(anchor_p90_values) / float(len(anchor_p90_values)),
        },
        "anchor_max": {
            "min": min(anchor_max_values),
            "max": max(anchor_max_values),
            "mean": sum(anchor_max_values) / float(len(anchor_max_values)),
        },
    }

    meta = {
        "input_jsonl": str(input_jsonl.as_posix()),
        "fallback_dataset_created": fallback_created,
        "n_samples_requested": n_samples_requested,
        "n_samples_written": len(samples_e0),
        "seed": args.seed,
        "max_units": args.max_units,
        "unitization_id": args.unitization_id,
        "balanced": bool(args.balanced),
        "selection": selection_meta,
        "label_count_0": selected_label_counts["count0"],
        "label_count_1": selected_label_counts["count1"],
        "label_count_null": selected_label_counts["countnull"],
        "embedding_backend_id": EMBEDDING_BACKEND_ID,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "vec8_methods": {
            "E0": {
                "vec8_postproc_id": E0_POSTPROC_ID,
                "vec8_postproc_aggregate": E0_AGGREGATE_ID,
                "steps_per_sample": 48,
            },
            "E1": {
                "vec8_postproc_id": E1_POSTPROC_ID,
                "projection_seed": args.seed,
                "vec8_postproc_matrix_hash_sha256": projection_hash,
                "steps_per_sample": "unit_count",
            },
            "E2": {
                "vec8_postproc_id": E2_POSTPROC_ID,
                "vec8_postproc_aggregate": E2_AGGREGATE_ID,
                "steps_per_sample": 48,
                "k": E2_K,
                "beta": E2_BETA,
                "root_count": e8_verify_info["root_count"],
                "roots_hash_sha256": e8_verify_info["roots_hash_sha256"],
                "root_verify_path": str(verify_written_path.as_posix()),
                "root_verify_sha256": e8_verify_sha256,
            },
        },
        "unit_count_stats": {
            "min": unit_count_min,
            "max": unit_count_max,
            "mean": unit_count_mean,
        },
        "e2_anchor_stats_summary": anchor_summary,
        "outputs": {
            "e0_input_json": str(out_e0.as_posix()),
            "e1_input_json": str(out_e1.as_posix()),
            "e2_input_json": str(out_e2.as_posix()),
            "e2_anchor_stats_csv": str(out_anchor_csv.as_posix()),
        },
    }
    write_json(out_meta, meta)

    print(f"Loaded records: {len(records)}")
    print(f"Selected samples: {len(selected)}")
    print(
        "Selected label counts: "
        f"0={selected_label_counts['count0']} "
        f"1={selected_label_counts['count1']} "
        f"null={selected_label_counts['countnull']}"
    )
    print(f"Wrote: {out_e0}")
    print(f"Wrote: {out_e1}")
    print(f"Wrote: {out_e2}")
    print(f"Wrote: {out_anchor_csv}")
    print(f"Wrote: {out_meta}")
    print(f"E1 matrix hash sha256: {projection_hash}")
    print(f"E8 roots verify file: {verify_written_path}")
    print(f"E8 roots verify sha256: {e8_verify_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
