"""GRAPH_XOR R1.1 one-shot capability-localization compiler and runner.

The file deliberately combines model-free case compilation, tokenizer-only audit,
the sequential BF16 scout, result validation, figure generation, and the single
Modal Sandbox launcher.  R1 v1.0 is imported read-only as an exact algebra/generator
library and is never mutated or reopened.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import datetime as dt
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else Path("/")
PLAN_PATH = HERE / "R1_1_CAPABILITY_LOCALIZATION.md"
RESULT_PATH = HERE / "r1_1_results.json"
FIGURE_PATH = HERE / "r1_1_capability_matrix.png"
V1_ROOT = Path(
    os.environ.get(
        "GRAPH_XOR_R1_V1_ROOT",
        str(REPO_ROOT / "workstream" / "local" / "graph_xor_r1"),
    )
)
V1_B0 = Path(os.environ.get("GRAPH_XOR_R1_B0_PATH", str(V1_ROOT / "b0")))
sys.path.insert(0, str(V1_B0))

from graph_xor_r1_b0.binding import package_sha256  # noqa: E402
from graph_xor_r1_b0.core import (  # noqa: E402
    LabeledWorld,
    apply_matrix,
    class_from_path_parities,
    generate_a2m_pairs,
    generate_decorated_theta,
    generate_unicyclic_world,
    labeled_theta_for_class,
    path_permutation_matrix,
    permute_theta_sample,
    query_answer,
    reorder_sample,
    theta_path_parities,
    unicyclic_class,
)
from graph_xor_r1_b0.renderers import render_world  # noqa: E402


SCHEMA_VERSION = "1.0"
CASE_SEED = 202_608_170_931
PARITY_LENGTH = 8
FORWARD_CEILING = 2_500
MODEL_ORDER = (
    ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
)
MODEL_FILES = {
    "Qwen/Qwen3-0.6B": (
        "config.json", "generation_config.json", "merges.txt", "model.safetensors",
        "tokenizer_config.json", "tokenizer.json", "vocab.json",
    ),
    "Qwen/Qwen3-1.7B": (
        "config.json", "generation_config.json", "merges.txt",
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "model.safetensors.index.json", "tokenizer_config.json", "tokenizer.json", "vocab.json",
    ),
    "Qwen/Qwen3-4B": (
        "config.json", "generation_config.json", "merges.txt",
        "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors", "model.safetensors.index.json",
        "tokenizer_config.json", "tokenizer.json", "vocab.json",
    ),
    "Qwen/Qwen3-8B": (
        "config.json", "generation_config.json", "merges.txt",
        "model-00001-of-00005.safetensors", "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors", "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors", "model.safetensors.index.json",
        "tokenizer_config.json", "tokenizer.json", "vocab.json",
    ),
}
TOKENIZER_FILES = (
    "config.json", "merges.txt", "tokenizer_config.json", "tokenizer.json", "vocab.json"
)
CODEBOOK_PAIRS = ((" A", " B"), (" K", " M"), (" R", " V"), (" X", " Z"))
# The exact Qwen3 non-thinking assistant prefix ends in a newline.  The
# tokenizer-only compile established that bare "0"/"1" (not space-prefixed
# variants) each append exactly one stable contextual token across all cases.
DIRECT_LITERALS = ("0", "1")
ALPHAS = ((1, 0), (0, 1), (1, 1))
CLASSES = ((0, 0), (0, 1), (1, 0), (1, 1))
PATH_PERMUTATIONS = tuple(itertools.permutations(range(3)))
SURFACE_ORDER = ("I0", "P0", "P1", "P2", "P3", "P4", "P5", "X0", "A2-M", "B")
DIAGNOSTIC_SURFACES = SURFACE_ORDER[:8]

EXPECTED_V1_SPEC_SHA256 = "fa6afd7502da3d4edd348fb626b9585344ee276cb8b3cef3d5b059e143a27e87"
EXPECTED_V1_MANIFEST_SHA256 = "bf02600c469c0f129e87a47fe9e6663a91ccd741827b0aa3eef5ce535fbfda14"
EXPECTED_V1_PACKAGE_SHA256 = "a8df19e8bc1e62fd284093ba9a24df0d905717eb89366dcedacb1f27bd9e6d8d"

TOKEN_BINDING: dict[str, Any] = {
    "status": "TOKENIZER_ONLY_COMPILE_PASS",
    "compile_aggregate_sha256": "8b8290c5adcb8b906bf0f87106d93aeaf17f8c7420d65ab801b7ac830c87e968",
    "case_ledger_sha256": "bb3b2ed439ff304628978f8632203809abfc31bd5d88a7623f5542156283469e",
    "direct_literal_token_ids": [15, 16],
    "common_tokenizer_files": {
        "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
    "models": {
        model_id: {
            "revision": revision,
            "config_sha256": config_sha256,
            "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
            "literal_token_ids": {
                " A": 362, " B": 425, " K": 730, " M": 386,
                " R": 431, " V": 647, " X": 1599, " Z": 1863,
                "0": 15, "1": 16,
            },
            "prefix_aggregate_sha256": "52b46925a927ef6c87c2c6cd27c8ca7e79f6c0481b52eefd2e1e53516993afcd",
            "maximum_prefix_token_positions": 541,
            "p4_p5_exact_token_count_and_multiset_match": True,
        }
        for model_id, revision, config_sha256 in (
            (
                "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca",
                "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
            ),
            (
                "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
            ),
            (
                "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c",
                "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
            ),
            (
                "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218",
                "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
            ),
        )
    },
}

RUNTIME_LOCK = {
    "python": "3.11.2",
    "torch": "2.7.1+cu126",
    "transformers": "5.15.0",
    "tokenizers": "0.22.2",
    "huggingface_hub": "1.27.0",
    "jinja2": "3.1.6",
    "accelerate": "1.14.0",
    "safetensors": "0.8.0",
    "Pillow": "11.3.0",
}
IMAGE_BASE = (
    "python:3.11.2-slim-bullseye"
    "@sha256:2f749ef90f54fd4b3c77cde78eec23ab5b8199d9ac84e4ced6ae523ef223ef7b"
)
MODAL_SDK_VERSION = "1.5.2"
MODAL_APP = "graph_xor-r1-1-capability-localization"
MODAL_VOLUME = "graph_xor-r1-1-scout"
GPU_REQUEST = "L40S"
CAMPAIGN_SECONDS_CEILING = 7_200
GPU_MEMORY_MIN_BYTES = 40 * 1024**3


class ScoutError(RuntimeError):
    """Fail-closed R1.1 compilation or execution error."""


@dataclass(frozen=True)
class Case:
    case_id: str
    surface: str
    pair_id: str
    semantic_answer: int
    prompt: str
    zero_literal: str
    one_literal: str
    candidate_pair_id: str
    strata: Mapping[str, str]
    metadata: Mapping[str, Any]


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_u64(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(_canonical_bytes(parts)).digest()[:8], "big")


def _rng(*parts: object) -> random.Random:
    return random.Random(_stable_u64("graph_xor-r1-1", CASE_SEED, *parts))


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def _case(
    surface: str,
    pair_id: str,
    semantic_answer: int,
    prompt: str,
    *,
    zero_literal: str = DIRECT_LITERALS[0],
    one_literal: str = DIRECT_LITERALS[1],
    candidate_pair_id: str = "direct-01",
    strata: Mapping[str, object] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Case:
    identity = {
        "surface": surface,
        "pair_id": pair_id,
        "semantic_answer": semantic_answer,
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "zero_literal": zero_literal,
        "one_literal": one_literal,
    }
    return Case(
        case_id=_sha256_bytes(_canonical_bytes(("graph_xor-r1-1-case", identity))),
        surface=surface,
        pair_id=pair_id,
        semantic_answer=semantic_answer,
        prompt=prompt,
        zero_literal=zero_literal,
        one_literal=one_literal,
        candidate_pair_id=candidate_pair_id,
        strata={str(key): str(value) for key, value in (strata or {}).items()},
        metadata=dict(metadata or {}),
    )


def _direct_prompt(body: str) -> str:
    return (
        "Solve one exact binary task without producing reasoning.\n"
        + body.rstrip()
        + "\nReturn the semantic bit itself. Output exactly 0 or 1 and nothing else.\nANSWER:"
    )


def _codebook_prompt(body: str, zero: str, one: str, entry_order: int) -> str:
    entries = [(0, zero), (1, one)]
    if entry_order == 1:
        entries.reverse()
    lines = "\n".join(f"{literal.strip()} means {semantic}." for semantic, literal in entries)
    return (
        body.rstrip()
        + "\nUse this codebook:\n"
        + lines
        + "\nOutput exactly one listed code and nothing else.\nANSWER:"
    )


def _relational_prompt(sample: LabeledWorld) -> str:
    return _direct_prompt(
        "Inspect the complete graph-XOR world.\n"
        + render_world(sample, "symbolic_xor")
        + "QUERY: Return the XOR parity around the unique cycle."
    )


def _relational_codebook_prompt(
    sample: LabeledWorld, zero: str, one: str, entry_order: int
) -> str:
    body = (
        "Inspect the complete graph-XOR world without producing reasoning.\n"
        + render_world(sample, "symbolic_xor")
        + "QUERY: Return the XOR parity around the unique cycle."
    )
    return _codebook_prompt(body, zero, one, entry_order)


def _cycle_pair(pair_index: int, semantic_answer: int) -> tuple[LabeledWorld, tuple[int, ...], tuple[int, ...]]:
    world = generate_unicyclic_world(
        cycle_length=PARITY_LENGTH,
        tree_edges=0,
        seed=_stable_u64("cycle-world", pair_index) % (2**31),
    )
    rng = _rng("cycle-bits", pair_index)
    prefix = [rng.randrange(2) for _ in range(PARITY_LENGTH - 1)]
    final = semantic_answer
    for bit in prefix:
        final ^= bit
    bits = tuple(prefix + [final])
    sample = LabeledWorld(world, bits)
    order = list(range(PARITY_LENGTH))
    _rng("cycle-order", pair_index).shuffle(order)
    if order == list(range(PARITY_LENGTH)):
        order = order[1:] + order[:1]
    return sample, bits, tuple(order)


def _build_i0() -> list[Case]:
    cases: list[Case] = []
    for pair_index, pair in enumerate(CODEBOOK_PAIRS):
        for mapping in (0, 1):
            zero, one = pair if mapping == 0 else pair[::-1]
            for entry_order in (0, 1):
                for repeat in range(2):
                    pair_id = f"I0:{pair_index}:{mapping}:{entry_order}:{repeat}"
                    for answer in (0, 1):
                        prompt = _codebook_prompt(
                            f"The correct semantic bit is {answer}.", zero, one, entry_order
                        )
                        cases.append(
                            _case(
                                "I0", pair_id, answer, prompt,
                                zero_literal=zero, one_literal=one,
                                candidate_pair_id=f"codebook-{pair_index}",
                                strata={
                                    "semantic_answer": answer,
                                    "mapping": mapping,
                                    "entry_order": entry_order,
                                    "token_pair": pair_index,
                                },
                                metadata={"entry_literals": [
                                    zero if entry_order == 0 else one,
                                    one if entry_order == 0 else zero,
                                ]},
                            )
                        )
    return cases


def _build_p0_p2() -> dict[str, list[Case]]:
    result = {surface: [] for surface in ("P0", "P1", "P2")}
    for pair_index in range(30):
        relation_seed = _rng("P2", pair_index).randrange(2)
        for answer in (0, 1):
            pair_id = f"basic:{pair_index}"
            result["P0"].append(
                _case("P0", pair_id, answer, _direct_prompt(f"The correct semantic bit is {answer}."),
                      strata={"semantic_answer": answer})
            )
            result["P1"].append(
                _case(
                    "P1", pair_id, answer,
                    _direct_prompt(f"A XOR B = {answer}. What is the displayed relation bit on edge A-B?"),
                    strata={"semantic_answer": answer},
                )
            )
            second = relation_seed ^ answer
            result["P2"].append(
                _case(
                    "P2", pair_id, answer,
                    _direct_prompt(
                        f"A XOR B = {relation_seed}.\nB XOR C = {second}.\nWhat is A XOR C?"
                    ),
                    strata={"semantic_answer": answer},
                    metadata={"edge_bits": [relation_seed, second]},
                )
            )
    return result


def _build_p3_p5() -> dict[str, list[Case]]:
    result = {surface: [] for surface in ("P3", "P4", "P5")}
    for pair_index in range(30):
        pair_id = f"cycle:{pair_index}"
        for answer in (0, 1):
            sample, bits, order = _cycle_pair(pair_index, answer)
            raw = " ".join(str(bit) for bit in bits)
            result["P3"].append(
                _case(
                    "P3", pair_id, answer,
                    _direct_prompt(f"BIT SEQUENCE (length {PARITY_LENGTH}): {raw}\nReturn the XOR of all bits."),
                    strata={"semantic_answer": answer, "parity_length": PARITY_LENGTH},
                    metadata={"bits": list(bits)},
                )
            )
            result["P4"].append(
                _case(
                    "P4", pair_id, answer, _relational_prompt(sample),
                    strata={"semantic_answer": answer, "parity_length": PARITY_LENGTH},
                    metadata={"bits": list(bits), "edge_order": list(range(PARITY_LENGTH))},
                )
            )
            shuffled = reorder_sample(sample, order)
            result["P5"].append(
                _case(
                    "P5", pair_id, answer, _relational_prompt(shuffled),
                    strata={"semantic_answer": answer, "parity_length": PARITY_LENGTH},
                    metadata={"bits": list(bits), "edge_order": list(order)},
                )
            )
    return result


def _build_x0() -> list[Case]:
    cases: list[Case] = []
    config_index = 0
    for pair_index, pair in enumerate(CODEBOOK_PAIRS):
        for mapping in (0, 1):
            zero, one = pair if mapping == 0 else pair[::-1]
            for entry_order in (0, 1):
                for repeat in range(2):
                    problem_index = config_index
                    config_index += 1
                    pair_id = f"X0:{problem_index}"
                    for answer in (0, 1):
                        sample, bits, order = _cycle_pair(problem_index, answer)
                        shuffled = reorder_sample(sample, order)
                        cases.append(
                            _case(
                                "X0", pair_id, answer,
                                _relational_codebook_prompt(shuffled, zero, one, entry_order),
                                zero_literal=zero, one_literal=one,
                                candidate_pair_id=f"codebook-{pair_index}",
                                strata={
                                    "semantic_answer": answer,
                                    "mapping": mapping,
                                    "entry_order": entry_order,
                                    "token_pair": pair_index,
                                },
                                metadata={
                                    "bits": list(bits), "edge_order": list(order),
                                    "entry_literals": [
                                        zero if entry_order == 0 else one,
                                        one if entry_order == 0 else zero,
                                    ],
                                },
                            )
                        )
    return cases


def _unique_worlds(family: str, count: int) -> list[Any]:
    worlds: list[Any] = []
    seen: set[str] = set()
    seed = 0
    while len(worlds) < count:
        actual_seed = _stable_u64("world", family, seed) % (2**31)
        if family == "A2-M":
            world = generate_unicyclic_world(cycle_length=8, tree_edges=8, seed=actual_seed)
        elif family == "B":
            world = generate_decorated_theta(
                path_internal_count=3, max_gadget_nodes=3, seed=actual_seed
            )
        else:
            raise ScoutError(f"unknown world family: {family}")
        seed += 1
        if world.group_hash not in seen:
            seen.add(world.group_hash)
            worlds.append(world)
        if seed > 10_000:
            raise ScoutError(f"could not find {count} unique {family} worlds")
    return worlds


def _build_a2m() -> list[Case]:
    cases: list[Case] = []
    for world_index, world in enumerate(_unique_worlds("A2-M", 30)):
        pair = generate_a2m_pairs(
            world, seed=_stable_u64("A2M-pair", world.group_hash)
        )[0]
        samples = sorted((pair.first, pair.second), key=unicyclic_class)
        if [unicyclic_class(sample) for sample in samples] != [0, 1]:
            raise ScoutError("A2-M pair lost its two semantic classes")
        pair_id = f"A2-M:{world.group_hash}"
        for answer, sample in enumerate(samples):
            prompt = _direct_prompt(
                "Inspect the complete graph-XOR world.\n"
                + render_world(sample, "symbolic_xor")
                + "QUERY: Return the XOR parity around the unique cycle; ignore every tree edge."
            )
            cases.append(
                _case(
                    "A2-M", pair_id, answer, prompt,
                    strata={"semantic_answer": answer},
                    metadata={"world_hash": world.group_hash, "pair_id_external": pair.pair_id},
                )
            )
    return cases


def _source_class(desired: Sequence[int], permutation: Sequence[int]) -> tuple[int, int]:
    matrix = path_permutation_matrix(permutation)
    for candidate in CLASSES:
        if apply_matrix(matrix, candidate) == tuple(desired):
            return candidate
    raise ScoutError("theta path action lacked an inverse class")


def _theta_query(alpha: Sequence[int]) -> str:
    if tuple(alpha) == (1, 0):
        return "Return the XOR parity on the cycle formed by theta paths P0 and P2."
    if tuple(alpha) == (0, 1):
        return "Return the XOR parity on the cycle formed by theta paths P1 and P2."
    return "Return the XOR parity on the cycle formed by theta paths P0 and P1."


def _build_b() -> list[Case]:
    cases: list[Case] = []
    for world_index, world in enumerate(_unique_worlds("B", 10)):
        for alpha_index, alpha in enumerate(ALPHAS):
            pair_id = f"B:{world.group_hash}:{alpha[0]}{alpha[1]}"
            for answer in (0, 1):
                options = [candidate for candidate in CLASSES if query_answer(candidate, alpha) == answer]
                desired_c = options[(world_index + alpha_index + answer) % 2]
                permutation = PATH_PERMUTATIONS[(world_index * 3 + alpha_index * 2 + answer) % 6]
                source_c = _source_class(desired_c, permutation)
                source = labeled_theta_for_class(
                    world, source_c,
                    gauge_seed=_stable_u64("B-gauge", world.group_hash, alpha, answer),
                )
                sample = permute_theta_sample(source, permutation)
                actual_c = class_from_path_parities(theta_path_parities(sample))
                if actual_c != desired_c or query_answer(actual_c, alpha) != answer:
                    raise ScoutError("direct B compiler violated alpha^T C")
                prompt = _direct_prompt(
                    "Inspect the complete graph-XOR theta world. A theta path Pk is the s-to-t "
                    "chain through nodes beginning pk:v; gadget nodes containing :g are not part "
                    "of a queried cycle.\n"
                    + render_world(sample, "symbolic_xor")
                    + "QUERY: " + _theta_query(alpha)
                )
                cases.append(
                    _case(
                        "B", pair_id, answer, prompt,
                        strata={"semantic_answer": answer, "alpha": f"{alpha[0]}{alpha[1]}"},
                        metadata={
                            "world_hash": world.group_hash,
                            "C": list(desired_c), "alpha": list(alpha),
                            "path_permutation": list(permutation),
                        },
                    )
                )
    return cases


def compile_cases() -> dict[str, list[Case]]:
    surfaces: dict[str, list[Case]] = {surface: [] for surface in SURFACE_ORDER}
    surfaces["I0"] = _build_i0()
    surfaces.update(_build_p0_p2())
    surfaces.update(_build_p3_p5())
    surfaces["X0"] = _build_x0()
    surfaces["A2-M"] = _build_a2m()
    surfaces["B"] = _build_b()
    _validate_cases(surfaces)
    return surfaces


def _validate_cases(surfaces: Mapping[str, Sequence[Case]]) -> None:
    expected = {"I0": 64, "X0": 64, **{surface: 60 for surface in SURFACE_ORDER if surface not in {"I0", "X0"}}}
    for surface, count in expected.items():
        if len(surfaces[surface]) != count:
            raise ScoutError(f"{surface} has {len(surfaces[surface])} cases, expected {count}")
        if Counter(case.semantic_answer for case in surfaces[surface]) != {0: count // 2, 1: count // 2}:
            raise ScoutError(f"{surface} semantic labels are not exactly balanced")
        by_pair = defaultdict(list)
        for case in surfaces[surface]:
            by_pair[case.pair_id].append(case.semantic_answer)
        if any(sorted(values) != [0, 1] for values in by_pair.values()):
            raise ScoutError(f"{surface} does not consist of exact y0/y1 matched pairs")
    if len({case.case_id for rows in surfaces.values() for case in rows}) != sum(map(len, surfaces.values())):
        raise ScoutError("case identifiers are not globally unique")
    for surface in ("I0", "X0"):
        keys = Counter(
            (case.strata["token_pair"], case.strata["mapping"], case.strata["entry_order"], case.semantic_answer)
            for case in surfaces[surface]
        )
        if set(keys.values()) != {2} or len(keys) != 32:
            raise ScoutError(f"{surface} codebook factorial is not exactly balanced")
    for pair_id in {case.pair_id for case in surfaces["P3"]}:
        for answer in (0, 1):
            records = {
                surface: next(case for case in surfaces[surface] if case.pair_id == pair_id and case.semantic_answer == answer)
                for surface in ("P3", "P4", "P5")
            }
            bits = [records[surface].metadata["bits"] for surface in records]
            if bits[0] != bits[1] or bits[1] != bits[2]:
                raise ScoutError("P3/P4/P5 lost their shared underlying bit problem")
    b_combos = Counter(
        (tuple(case.metadata["C"]), tuple(case.metadata["alpha"])) for case in surfaces["B"]
    )
    if set(b_combos) != set(itertools.product(CLASSES, ALPHAS)) or set(b_combos.values()) != {5}:
        raise ScoutError("direct B does not realize every C×alpha combination five times")
    if len({case.metadata["world_hash"] for case in surfaces["B"]}) != 10:
        raise ScoutError("direct B lacks ten independent decorated-theta morphologies")
    if sum(len(surfaces[surface]) for surface in DIAGNOSTIC_SURFACES) != 488:
        raise ScoutError("per-model diagnostic forward count is not 488")
    if 4 * 488 + 4 * 120 > FORWARD_CEILING:
        raise ScoutError("frozen worst-case forward allocation exceeds its ceiling")


def _public_case(case: Case) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "surface": case.surface,
        "pair_id": case.pair_id,
        "semantic_answer": case.semantic_answer,
        "prompt_sha256": _sha256_bytes(case.prompt.encode("utf-8")),
        "zero_literal": case.zero_literal,
        "one_literal": case.one_literal,
        "candidate_pair_id": case.candidate_pair_id,
        "strata": dict(case.strata),
        "metadata": dict(case.metadata),
    }


def _download_tokenizers(cache_root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    from huggingface_hub import snapshot_download

    summaries: dict[str, Any] = {}
    snapshots: dict[str, Path] = {}
    for model_id, revision in MODEL_ORDER:
        snapshot = Path(snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=list(TOKENIZER_FILES),
            cache_dir=str(cache_root),
        ))
        files = sorted(path for path in snapshot.rglob("*") if path.is_file())
        names = {path.relative_to(snapshot).as_posix() for path in files}
        if names != set(TOKENIZER_FILES):
            raise ScoutError(f"{model_id} tokenizer inventory mismatch: {sorted(names)}")
        summaries[model_id] = {
            "revision": revision,
            "files": [
                {
                    "name": path.relative_to(snapshot).as_posix(),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in files
            ],
        }
        snapshots[model_id] = snapshot
    return summaries, snapshots


def _chat_prefix(tokenizer: Any, prompt: str) -> tuple[str, list[int]]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(value) for value in ids]
    if tokenizer(text, add_special_tokens=False)["input_ids"] != ids:
        raise ScoutError("chat-template text and normative token IDs disagree")
    return str(text), ids


def _appended_token_id(tokenizer: Any, text: str, ids: Sequence[int], literal: str) -> int:
    appended = tokenizer(text + literal, add_special_tokens=False)["input_ids"]
    if appended[: len(ids)] != list(ids) or len(appended) != len(ids) + 1:
        raise ScoutError(f"literal {literal!r} does not append exactly one contextual token")
    token_id = int(appended[-1])
    if token_id in set(tokenizer.all_special_ids):
        raise ScoutError(f"literal {literal!r} resolves to a special token")
    return token_id


def tokenizer_only_compile(cache_root: Path) -> dict[str, Any]:
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        raise ScoutError("tokenizer-only compile must run in an environment without PyTorch")
    from transformers import AutoTokenizer

    cases = compile_cases()
    acquisition, snapshots = _download_tokenizers(cache_root)
    model_results: dict[str, Any] = {}
    for model_id, revision in MODEL_ORDER:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshots[model_id], use_fast=True, local_files_only=True, trust_remote_code=False
        )
        literal_ids: dict[str, set[int]] = defaultdict(set)
        prefix_records: list[dict[str, Any]] = []
        p4_tokens: dict[tuple[str, int], list[int]] = {}
        p5_tokens: dict[tuple[str, int], list[int]] = {}
        maximum_positions = 0
        for surface in SURFACE_ORDER:
            for case in cases[surface]:
                text, ids = _chat_prefix(tokenizer, case.prompt)
                maximum_positions = max(maximum_positions, len(ids))
                for literal in (case.zero_literal, case.one_literal):
                    literal_ids[literal].add(_appended_token_id(tokenizer, text, ids, literal))
                prefix_records.append({
                    "case_id": case.case_id,
                    "token_ids_sha256": _sha256_bytes(_canonical_bytes(ids)),
                    "token_positions": len(ids),
                })
                key = (case.pair_id, case.semantic_answer)
                if surface == "P4":
                    p4_tokens[key] = ids
                elif surface == "P5":
                    p5_tokens[key] = ids
        if set(p4_tokens) != set(p5_tokens):
            raise ScoutError("P4/P5 token audit keys differ")
        for key in p4_tokens:
            if len(p4_tokens[key]) != len(p5_tokens[key]) or Counter(p4_tokens[key]) != Counter(p5_tokens[key]):
                raise ScoutError(f"P4/P5 token matching failed for {key}")
        if any(len(values) != 1 for values in literal_ids.values()):
            raise ScoutError(f"{model_id} has context-dependent candidate token IDs")
        model_results[model_id] = {
            "revision": revision,
            "chat_template_sha256": _sha256_bytes(str(tokenizer.chat_template).encode("utf-8")),
            "literal_token_ids": {literal: next(iter(values)) for literal, values in sorted(literal_ids.items())},
            "prefix_aggregate_sha256": _sha256_bytes(_canonical_bytes(prefix_records)),
            "maximum_prefix_token_positions": maximum_positions,
            "cases_audited": len(prefix_records),
            "p4_p5_exact_token_count_and_multiset_match": True,
        }
    direct_sets = {tuple(result["literal_token_ids"][literal] for literal in DIRECT_LITERALS) for result in model_results.values()}
    if len(direct_sets) != 1:
        raise ScoutError("direct 0/1 token IDs differ across the model family")
    aggregate = {
        "status": "TOKENIZER_ONLY_COMPILE_PASS",
        "models": model_results,
        "acquisition": acquisition,
        "case_ledger_sha256": _sha256_bytes(_canonical_bytes({
            surface: [_public_case(case) for case in cases[surface]] for surface in SURFACE_ORDER
        })),
        "case_counts": {surface: len(cases[surface]) for surface in SURFACE_ORDER},
        "direct_literal_token_ids": list(next(iter(direct_sets))),
        "runtime": {
            key: importlib.metadata.version(package)
            for key, package in (
                ("transformers", "transformers"), ("tokenizers", "tokenizers"),
                ("huggingface_hub", "huggingface-hub"), ("jinja2", "jinja2"),
            )
        },
    }
    aggregate["aggregate_sha256"] = _sha256_bytes(_canonical_bytes(aggregate))
    return aggregate


def _auc(records: Sequence[Mapping[str, Any]]) -> float:
    positives = [float(row["semantic_score"]) for row in records if row["semantic_answer"] == 1]
    negatives = [float(row["semantic_score"]) for row in records if row["semantic_answer"] == 0]
    points = 0.0
    for positive in positives:
        for negative in negatives:
            points += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return points / (len(positives) * len(negatives))


def _accuracy(records: Sequence[Mapping[str, Any]]) -> float:
    points = 0.0
    for row in records:
        score = float(row["semantic_score"])
        answer = int(row["semantic_answer"])
        points += 0.5 if score == 0 else float((score > 0) == (answer == 1))
    return points / len(records)


def _paired_direction(records: Sequence[Mapping[str, Any]]) -> float:
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row in records:
        grouped[str(row["pair_id"])][int(row["semantic_answer"])] = float(row["semantic_score"])
    if any(set(values) != {0, 1} for values in grouped.values()):
        raise ScoutError("metric records lost exact matched pairs")
    points = [1.0 if values[1] > values[0] else 0.5 if values[1] == values[0] else 0.0 for values in grouped.values()]
    return sum(points) / len(points)


def classify_surface(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accuracy = _accuracy(records)
    auc = _auc(records)
    paired = _paired_direction(records)
    strata: dict[str, dict[str, float]] = {}
    stratum_keys = sorted({key for row in records for key in row["strata"]})
    for key in stratum_keys:
        values = sorted({str(row["strata"][key]) for row in records})
        if len(values) > 1:
            strata[key] = {
                value: _accuracy([row for row in records if str(row["strata"][key]) == value])
                for value in values
            }
    minimum_stratum = min(
        (value for table in strata.values() for value in table.values()), default=accuracy
    )
    semantic_counts = Counter(int(row["semantic_prediction"]) for row in records)
    semantic_collapse = max(semantic_counts.values()) / len(records) >= 0.95
    by_candidate_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_candidate_pair[str(row["candidate_pair_id"])].append(row)
    physical_ratio = max(
        max(Counter(str(row["physical_choice_literal"]) for row in rows).values()) / len(rows)
        for rows in by_candidate_pair.values()
    )
    physical_collapse = physical_ratio >= 0.95
    entry_records = [row for row in records if row.get("entry_position") is not None]
    if entry_records:
        entry_ratio = max(Counter(int(row["entry_position"]) for row in entry_records).values()) / len(entry_records)
        entry_collapse = entry_ratio >= 0.95
    else:
        entry_ratio = None
        entry_collapse = False
    inversion = auc <= 0.10 and paired <= 0.10
    inversion_warning = (auc <= 0.10) != (paired <= 0.10)
    pass_flag = (
        accuracy >= 0.75 and paired >= 0.75 and minimum_stratum >= 0.625
        and not semantic_collapse and not physical_collapse
        and not entry_collapse and not inversion
    )
    if pass_flag:
        classification = "BEHAVIOR_PASS"
    elif auc >= 0.75 or paired >= 0.75:
        classification = "SCORE_SIGNAL_ONLY"
    else:
        classification = "NO_DETECTED_SIGNAL"
    mu = {
        answer: sum(float(row["semantic_score"]) for row in records if row["semantic_answer"] == answer)
        / sum(row["semantic_answer"] == answer for row in records)
        for answer in (0, 1)
    }
    return {
        "classification": classification,
        "semantic_accuracy": accuracy,
        "semantic_score_auc": auc,
        "paired_directional_consistency": paired,
        "stratum_accuracy": strata,
        "minimum_applicable_stratum_accuracy": minimum_stratum,
        "semantic_score_mean_by_label": {str(key): value for key, value in mu.items()},
        "semantic_score_offset": (mu[1] + mu[0]) / 2,
        "semantic_score_separation": (mu[1] - mu[0]) / 2,
        "semantic_prediction_rate": {
            str(key): value / len(records) for key, value in sorted(semantic_counts.items())
        },
        "collapse": {
            "semantic_label": semantic_collapse,
            "physical_token_pair_conditioned": physical_collapse,
            "physical_token_max_ratio": physical_ratio,
            "entry_position": entry_collapse,
            "entry_position_max_ratio": entry_ratio,
            "systematic_inversion": inversion,
            "inversion_warning": inversion_warning,
        },
    }


def _runtime_environment() -> dict[str, Any]:
    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "jinja2": importlib.metadata.version("jinja2"),
        "accelerate": importlib.metadata.version("accelerate"),
        "safetensors": importlib.metadata.version("safetensors"),
        "Pillow": importlib.metadata.version("Pillow"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_device_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        ),
        "cuda_driver": (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True, capture_output=True, text=True,
            ).stdout.strip().splitlines()[0]
            if torch.cuda.is_available() else None
        ),
    }


def _verify_runtime() -> dict[str, Any]:
    import torch

    runtime = _runtime_environment()
    for key, expected in RUNTIME_LOCK.items():
        if runtime.get(key) != expected:
            raise ScoutError(f"runtime drift for {key}: {runtime.get(key)!r} != {expected!r}")
    if not runtime["cuda_available"] or "L40S" not in str(runtime["cuda_device_name"]).upper():
        raise ScoutError(f"campaign did not receive the frozen L40S GPU: {runtime['cuda_device_name']}")
    if int(runtime["cuda_device_total_memory_bytes"]) < GPU_MEMORY_MIN_BYTES:
        raise ScoutError("campaign GPU has less than the frozen 40 GiB minimum")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(CASE_SEED)
    torch.cuda.manual_seed_all(CASE_SEED)
    return runtime


def _verify_v1_immutable() -> dict[str, str]:
    spec = V1_ROOT / "GRAPH_XOR_R1_GLOBAL_OBSTRUCTION_SPEC_v1.0.md"
    manifest_path = V1_ROOT / "study_manifest.json"
    actual = {
        "spec_sha256": _sha256_file(spec),
        "manifest_sha256": _sha256_file(manifest_path),
        "b0_package_sha256": package_sha256(),
    }
    expected = {
        "spec_sha256": EXPECTED_V1_SPEC_SHA256,
        "manifest_sha256": EXPECTED_V1_MANIFEST_SHA256,
        "b0_package_sha256": EXPECTED_V1_PACKAGE_SHA256,
    }
    if actual != expected:
        raise ScoutError(f"immutable R1 v1.0 binding drifted: {actual}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"]["current_stage"] != "CLOSED_AT_C0" or manifest["status"]["s0"] != "S0_SELECTION_FAIL":
        raise ScoutError("R1 v1.0 is not in its frozen closeout state")
    if any(manifest["status"].get(key, False) for key in (
        "model_execution_authorized", "activation_extraction_authorized",
        "alignment_search_authorized", "semantic_holdout_opened",
    )):
        raise ScoutError("R1 v1.0 has an open scientific permission")
    return actual


def _acquire_model(model_id: str, revision: str, cache_root: Path) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=model_id, revision=revision,
        allow_patterns=list(MODEL_FILES[model_id]), cache_dir=str(cache_root),
    ))
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    names = {path.relative_to(snapshot).as_posix() for path in files}
    if names != set(MODEL_FILES[model_id]):
        raise ScoutError(f"{model_id} full snapshot inventory mismatch")
    return snapshot, {
        "revision": revision,
        "files": [
            {"name": path.relative_to(snapshot).as_posix(), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            for path in files
        ],
    }


def _verify_token_binding(model_id: str, tokenizer: Any, cases: Mapping[str, Sequence[Case]]) -> None:
    expected = TOKEN_BINDING["models"][model_id]
    if _sha256_bytes(str(tokenizer.chat_template).encode("utf-8")) != expected["chat_template_sha256"]:
        raise ScoutError(f"{model_id} chat template differs from compile binding")
    literal_ids: dict[str, set[int]] = defaultdict(set)
    prefix_records: list[dict[str, Any]] = []
    for surface in SURFACE_ORDER:
        for case in cases[surface]:
            text, ids = _chat_prefix(tokenizer, case.prompt)
            for literal in (case.zero_literal, case.one_literal):
                literal_ids[literal].add(_appended_token_id(tokenizer, text, ids, literal))
            prefix_records.append({
                "case_id": case.case_id,
                "token_ids_sha256": _sha256_bytes(_canonical_bytes(ids)),
                "token_positions": len(ids),
            })
    actual_ids = {literal: next(iter(values)) for literal, values in literal_ids.items() if len(values) == 1}
    if actual_ids != expected["literal_token_ids"]:
        raise ScoutError(f"{model_id} candidate token binding drift")
    if _sha256_bytes(_canonical_bytes(prefix_records)) != expected["prefix_aggregate_sha256"]:
        raise ScoutError(f"{model_id} prefix ledger drift")


def _score_case(model: Any, tokenizer: Any, case: Case, device: str) -> dict[str, Any]:
    import torch

    text, ids = _chat_prefix(tokenizer, case.prompt)
    zero_id = _appended_token_id(tokenizer, text, ids, case.zero_literal)
    one_id = _appended_token_id(tokenizer, text, ids, case.one_literal)
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            input_ids=tensor, use_cache=False,
            output_hidden_states=False, output_attentions=False, return_dict=True,
        )
        logits = outputs.logits[0, -1, [zero_id, one_id]].float().cpu().tolist()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    del outputs, tensor
    zero_logit, one_logit = map(float, logits)
    score = one_logit - zero_logit
    if score > 0:
        semantic_prediction = 1
        chosen_id, chosen_literal = one_id, case.one_literal
    elif score < 0:
        semantic_prediction = 0
        chosen_id, chosen_literal = zero_id, case.zero_literal
    elif zero_id <= one_id:
        semantic_prediction = 0
        chosen_id, chosen_literal = zero_id, case.zero_literal
    else:
        semantic_prediction = 1
        chosen_id, chosen_literal = one_id, case.one_literal
    entry_literals = case.metadata.get("entry_literals")
    entry_position = None
    if entry_literals:
        entry_position = list(entry_literals).index(chosen_literal)
    return {
        "case_id": case.case_id,
        "surface": case.surface,
        "pair_id": case.pair_id,
        "semantic_answer": case.semantic_answer,
        "semantic_score": score,
        "semantic_prediction": semantic_prediction,
        "zero_logit": zero_logit,
        "one_logit": one_logit,
        "zero_token_id": zero_id,
        "one_token_id": one_id,
        "physical_choice_token_id": chosen_id,
        "physical_choice_literal": chosen_literal,
        "candidate_pair_id": case.candidate_pair_id,
        "entry_position": entry_position,
        "strata": dict(case.strata),
        "prompt_sha256": _sha256_bytes(case.prompt.encode("utf-8")),
        "prefix_token_ids_sha256": _sha256_bytes(_canonical_bytes(ids)),
        "token_positions": len(ids),
        "forward_seconds": elapsed,
    }


def _new_result(plan_sha: str, runner_sha: str, runtime: Mapping[str, Any], v1: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "program_id": "GRAPH_XOR_R1_1_CAPABILITY_LOCALIZATION",
        "status": "SCOUT_IN_PROGRESS",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan_sha256": plan_sha,
        "runner_sha256": runner_sha,
        "case_generation_seed": CASE_SEED,
        "forward_ceiling": FORWARD_CEILING,
        "forwards_completed": 0,
        "runtime_environment": dict(runtime),
        "v1_immutable_binding": dict(v1),
        "token_binding": TOKEN_BINDING,
        "model_order": [model_id for model_id, _ in MODEL_ORDER],
        "models": {
            model_id: {
                "revision": revision, "status": "UNOPENED", "surfaces": {},
            }
            for model_id, revision in MODEL_ORDER
        },
        "stop_reason": None,
    }


def _render_figure(result: Mapping[str, Any], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rows = [model_id for model_id, _ in MODEL_ORDER]
    columns = list(SURFACE_ORDER)
    left, top, cell_w, cell_h = 245, 88, 82, 54
    width, height = left + cell_w * len(columns) + 24, top + cell_h * len(rows) + 72
    image = Image.new("RGB", (width, height), "#111318")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = {
        "BEHAVIOR_PASS": "#35c46a",
        "SCORE_SIGNAL_ONLY": "#e5a93d",
        "NO_DETECTED_SIGNAL": "#d45454",
        "UNOPENED": "#414752",
    }
    draw.text((18, 18), "GRAPH_XOR R1.1 - Capability Localization", fill="white", font=font)
    draw.text((18, 40), f"status: {result.get('status')} | forwards: {result.get('forwards_completed')}", fill="#b7bdc8", font=font)
    for column, surface in enumerate(columns):
        x = left + column * cell_w
        draw.text((x + 8, top - 28), surface, fill="#dce2ed", font=font)
    for row, model_id in enumerate(rows):
        y = top + row * cell_h
        draw.text((18, y + 18), model_id, fill="#dce2ed", font=font)
        model = result["models"][model_id]
        for column, surface in enumerate(columns):
            classification = model.get("surfaces", {}).get(surface, {}).get("metrics", {}).get("classification", "UNOPENED")
            x = left + column * cell_w
            draw.rectangle((x, y, x + cell_w - 6, y + cell_h - 6), fill=colors[classification], outline="#858b95")
            label = {"BEHAVIOR_PASS": "PASS", "SCORE_SIGNAL_ONLY": "SIGNAL", "NO_DETECTED_SIGNAL": "NONE", "UNOPENED": "-"}[classification]
            draw.text((x + 10, y + 18), label, fill="white", font=font)
    legend_y = top + cell_h * len(rows) + 18
    draw.text((18, legend_y), "green PASS  |  amber score signal only  |  red no detected signal  |  gray unopened", fill="#b7bdc8", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def execute_scout(
    *, plan_path: Path, result_path: Path, figure_path: Path,
    expected_plan_sha: str, expected_runner_sha: str, cache_root: Path,
) -> dict[str, Any]:
    actual_plan_sha = _sha256_file(plan_path)
    actual_runner_sha = _sha256_file(Path(__file__))
    if actual_plan_sha != expected_plan_sha or actual_runner_sha != expected_runner_sha:
        raise ScoutError("plan or runner SHA differs from the authorized binding")
    if TOKEN_BINDING.get("status") != "TOKENIZER_ONLY_COMPILE_PASS":
        raise ScoutError("runner lacks its frozen tokenizer binding")
    runtime = _verify_runtime()
    v1 = _verify_v1_immutable()
    cases = compile_cases()
    ledger_sha = _sha256_bytes(_canonical_bytes({
        surface: [_public_case(case) for case in cases[surface]] for surface in SURFACE_ORDER
    }))
    if ledger_sha != TOKEN_BINDING["case_ledger_sha256"]:
        raise ScoutError("case ledger differs from the tokenizer-only compile")
    result = _new_result(actual_plan_sha, actual_runner_sha, runtime, v1)
    result["case_ledger_sha256"] = ledger_sha
    _atomic_json(result_path, result)
    _render_figure(result, figure_path)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        for model_id, revision in MODEL_ORDER:
            model_entry = result["models"][model_id]
            model_entry["status"] = "ACQUIRING"
            _atomic_json(result_path, result)
            print(f"R1.1 opening {model_id}: acquiring exact revision", flush=True)
            snapshot, inventory = _acquire_model(model_id, revision, cache_root)
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot, use_fast=True, local_files_only=True, trust_remote_code=False
            )
            _verify_token_binding(model_id, tokenizer, cases)
            model_entry["snapshot_inventory"] = inventory
            model_entry["status"] = "LOADING"
            _atomic_json(result_path, result)
            print(f"R1.1 opening {model_id}: loading BF16 on cuda:0", flush=True)
            load_started = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                snapshot,
                dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=False,
                attn_implementation="eager",
            )
            model.eval()
            torch.cuda.synchronize()
            model_entry["model_load_seconds"] = time.perf_counter() - load_started
            model_entry["status"] = "RUNNING"
            for surface in DIAGNOSTIC_SURFACES:
                records = [_score_case(model, tokenizer, case, "cuda:0") for case in cases[surface]]
                result["forwards_completed"] += len(records)
                if result["forwards_completed"] > FORWARD_CEILING:
                    raise ScoutError("forward ceiling exceeded")
                model_entry["surfaces"][surface] = {
                    "status": "OPENED_COMPLETE",
                    "cases": len(records),
                    "records": records,
                    "metrics": classify_surface(records),
                }
                print(
                    f"R1.1 {model_id} {surface}: "
                    f"{model_entry['surfaces'][surface]['metrics']['classification']}",
                    flush=True,
                )
                _atomic_json(result_path, result)
                _render_figure(result, figure_path)
            if model_entry["surfaces"]["P5"]["metrics"]["classification"] == "BEHAVIOR_PASS":
                records = [_score_case(model, tokenizer, case, "cuda:0") for case in cases["A2-M"]]
                result["forwards_completed"] += len(records)
                model_entry["surfaces"]["A2-M"] = {
                    "status": "OPENED_COMPLETE", "cases": len(records),
                    "records": records, "metrics": classify_surface(records),
                }
                print(
                    f"R1.1 {model_id} A2-M: "
                    f"{model_entry['surfaces']['A2-M']['metrics']['classification']}",
                    flush=True,
                )
                _atomic_json(result_path, result)
                _render_figure(result, figure_path)
                if model_entry["surfaces"]["A2-M"]["metrics"]["classification"] == "BEHAVIOR_PASS":
                    records = [_score_case(model, tokenizer, case, "cuda:0") for case in cases["B"]]
                    result["forwards_completed"] += len(records)
                    model_entry["surfaces"]["B"] = {
                        "status": "OPENED_COMPLETE", "cases": len(records),
                        "records": records, "metrics": classify_surface(records),
                    }
                    print(
                        f"R1.1 {model_id} B: "
                        f"{model_entry['surfaces']['B']['metrics']['classification']}",
                        flush=True,
                    )
                    _atomic_json(result_path, result)
                    _render_figure(result, figure_path)
            model_entry["status"] = "COMPLETE"
            passed_b = model_entry["surfaces"].get("B", {}).get("metrics", {}).get("classification") == "BEHAVIOR_PASS"
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            if passed_b:
                result["status"] = "SCOUT_PASS_DIRECT_B"
                result["stop_reason"] = f"first direct B BEHAVIOR_PASS at {model_id}; larger models remain unopened"
                break
        else:
            result["status"] = "SCOUT_CLOSE_NO_DIRECT_B_THROUGH_8B"
            result["stop_reason"] = "Qwen3 0.6B–8B zero-shot/non-thinking/direct-semantic graph-XOR path closed"
        result["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        result["unopened_models"] = [
            model_id for model_id, _ in MODEL_ORDER if result["models"][model_id]["status"] == "UNOPENED"
        ]
        _atomic_json(result_path, result)
        _render_figure(result, figure_path)
        return result
    except Exception as error:
        result["status"] = "OPERATIONAL_ABORT"
        result["stop_reason"] = f"{type(error).__name__}: {error}"
        result["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _atomic_json(result_path, result)
        _render_figure(result, figure_path)
        raise


def _modal_image(modal: Any) -> Any:
    packages = [
        "transformers==5.15.0", "tokenizers==0.22.2", "huggingface_hub==1.27.0",
        "jinja2==3.1.6", "accelerate==1.14.0", "safetensors==0.8.0",
        "pillow==11.3.0", "psutil==7.2.2",
    ]
    return (
        modal.Image.from_registry(IMAGE_BASE)
        .apt_install("libgomp1")
        .pip_install("torch==2.7.1", extra_index_url="https://download.pytorch.org/whl/cu126")
        .pip_install(*packages)
        .add_local_dir(HERE, "/opt/graph_xor_r1_1", copy=True, ignore=["r1_1_results.json", "r1_1_capability_matrix.png", "**/__pycache__/**"])
        .add_local_dir(V1_ROOT, "/opt/graph_xor_r1_v1", copy=True, ignore=["b0/rust/target/**", "**/__pycache__/**"])
    )


def _stream_process(process: Any) -> tuple[int, str, str]:
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    def pump(stream: Any, destination: list[str], prefix: str) -> None:
        for line in stream:
            destination.append(line)
            print(f"[{prefix}] {line}", end="", flush=True)
    threads = [
        threading.Thread(target=pump, args=(process.stdout, stdout_parts, "remote"), daemon=True),
        threading.Thread(target=pump, args=(process.stderr, stderr_parts, "remote-error"), daemon=True),
    ]
    for thread in threads:
        thread.start()
    code = process.wait()
    for thread in threads:
        thread.join()
    return code, "".join(stdout_parts), "".join(stderr_parts)


def run_modal_campaign(plan_path: Path, result_path: Path, figure_path: Path) -> dict[str, Any]:
    installed = importlib.metadata.version("modal")
    if installed != MODAL_SDK_VERSION:
        raise ScoutError(f"Modal SDK drift: {installed} != {MODAL_SDK_VERSION}")
    import modal

    plan_sha = _sha256_file(plan_path)
    runner_sha = _sha256_file(Path(__file__))
    plan_text = plan_path.read_text(encoding="utf-8")
    if f"RUNNER_SHA256: {runner_sha}" not in plan_text:
        raise ScoutError("one-page plan is not bound to this runner")
    if "TOKENIZER_BINDING: TO_BE_COMPILED" in plan_text or TOKEN_BINDING.get("status") != "TOKENIZER_ONLY_COMPILE_PASS":
        raise ScoutError("one-page plan or runner lacks the final tokenizer binding")
    app = modal.App.lookup(MODAL_APP, create_if_missing=True)
    volume = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True, version=2)
    volume.hydrate()
    image = _modal_image(modal)
    with modal.enable_output():
        image = image.build(app)
    sandbox = modal.Sandbox.create(
        app=app, image=image, gpu=GPU_REQUEST, cpu=4.0, memory=32768,
        timeout=CAMPAIGN_SECONDS_CEILING, workdir="/opt/graph_xor_r1_1",
        volumes={"/vol": volume},
        env={
            "GRAPH_XOR_R1_B0_PATH": "/opt/graph_xor_r1_v1/b0",
            "GRAPH_XOR_R1_V1_ROOT": "/opt/graph_xor_r1_v1",
            "HF_HOME": "/vol/hf-home",
            "PYTHONUTF8": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        tags={"program": "graph_xor-r1-1", "type": "capability-localization", "campaign": "one-shot"},
    )
    run_id = "r1-1-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_root = f"/vol/r1_1/{run_id}"
    remote_result = f"{remote_root}/r1_1_results.json"
    remote_figure = f"{remote_root}/r1_1_capability_matrix.png"
    try:
        process = sandbox.exec(
            "python", "/opt/graph_xor_r1_1/run_r1_1.py", "execute",
            "--plan", "/opt/graph_xor_r1_1/R1_1_CAPABILITY_LOCALIZATION.md",
            "--result", remote_result, "--figure", remote_figure,
            "--expected-plan-sha", plan_sha, "--expected-runner-sha", runner_sha,
            "--cache-root", "/vol/hf-home",
            timeout=CAMPAIGN_SECONDS_CEILING - 180,
            workdir="/opt/graph_xor_r1_1",
        )
        code, _, stderr = _stream_process(process)
        sync = sandbox.exec("sync", "/vol", timeout=120)
        sync.wait()
        if code != 0:
            try:
                sandbox.filesystem.copy_to_local(remote_result, str(result_path))
                sandbox.filesystem.copy_to_local(remote_figure, str(figure_path))
            except Exception:
                pass
            raise ScoutError(f"remote campaign exited {code}: {stderr[-4000:]}")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        sandbox.filesystem.copy_to_local(remote_result, str(result_path))
        sandbox.filesystem.copy_to_local(remote_figure, str(figure_path))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["plan_sha256"] != plan_sha or result["runner_sha256"] != runner_sha:
            raise ScoutError("downloaded result lost its plan/runner binding")
        return result
    finally:
        try:
            sandbox.terminate(wait=False)
        finally:
            try:
                sandbox.detach()
            except Exception:
                pass


def self_test() -> dict[str, Any]:
    cases = compile_cases()
    synthetic: list[dict[str, Any]] = []
    for case in cases["P0"]:
        score = 1.0 if case.semantic_answer == 1 else -1.0
        synthetic.append({
            "pair_id": case.pair_id,
            "semantic_answer": case.semantic_answer,
            "semantic_score": score,
            "semantic_prediction": case.semantic_answer,
            "candidate_pair_id": case.candidate_pair_id,
            "physical_choice_literal": case.one_literal if case.semantic_answer else case.zero_literal,
            "entry_position": None,
            "strata": dict(case.strata),
        })
    metrics = classify_surface(synthetic)
    if metrics["classification"] != "BEHAVIOR_PASS":
        raise ScoutError("perfect synthetic score failed BEHAVIOR_PASS")
    if metrics["semantic_score_offset"] != 0.0 or metrics["semantic_score_separation"] != 1.0:
        raise ScoutError("offset/separation formulas are incorrect")
    return {
        "status": "SELF_TEST_PASS",
        "case_counts": {surface: len(rows) for surface, rows in cases.items()},
        "case_ledger_sha256": _sha256_bytes(_canonical_bytes({
            surface: [_public_case(case) for case in rows] for surface, rows in cases.items()
        })),
        "perfect_synthetic_metrics": metrics,
        "worst_case_scored_forwards": 4 * 488 + 4 * 120,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    compile_parser = sub.add_parser("compile-tokenizers")
    compile_parser.add_argument("--cache-root", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--plan", type=Path, required=True)
    execute_parser.add_argument("--result", type=Path, required=True)
    execute_parser.add_argument("--figure", type=Path, required=True)
    execute_parser.add_argument("--expected-plan-sha", required=True)
    execute_parser.add_argument("--expected-runner-sha", required=True)
    execute_parser.add_argument("--cache-root", type=Path, required=True)
    modal_parser = sub.add_parser("modal-campaign")
    modal_parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    modal_parser.add_argument("--result", type=Path, default=RESULT_PATH)
    modal_parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    args = parser.parse_args(argv)
    if args.command == "self-test":
        print(json.dumps(self_test(), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "compile-tokenizers":
        report = tokenizer_only_compile(args.cache_root)
        _atomic_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "execute":
        result = execute_scout(
            plan_path=args.plan, result_path=args.result, figure_path=args.figure,
            expected_plan_sha=args.expected_plan_sha,
            expected_runner_sha=args.expected_runner_sha, cache_root=args.cache_root,
        )
        print(json.dumps({
            "status": result["status"], "forwards_completed": result["forwards_completed"],
            "stop_reason": result["stop_reason"],
        }, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        result = run_modal_campaign(args.plan, args.result, args.figure)
        print(json.dumps({
            "status": result["status"], "forwards_completed": result["forwards_completed"],
            "stop_reason": result["stop_reason"],
        }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
