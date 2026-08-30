"""GRAPH_XOR R1.2 one-shot input-output-only ICL formation scout.

R1 v1.0 and R1.1 are imported read-only.  This single file compiles the bounded
demo/target ledger, audits the exact tokenization, executes the sequential BF16
campaign, validates the result, and renders the capability matrix.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import datetime as dt
import difflib
import gc
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import tempfile
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
R11_ROOT = Path(os.environ.get("GRAPH_XOR_R11_ROOT", str(HERE.parent / "graph_xor_r1_1")))
V1_ROOT = Path(os.environ.get("GRAPH_XOR_R1_V1_ROOT", str(HERE.parent / "graph_xor_r1")))
os.environ.setdefault("GRAPH_XOR_R1_V1_ROOT", str(V1_ROOT))
os.environ.setdefault("GRAPH_XOR_R1_B0_PATH", str(V1_ROOT / "b0"))

_r11_spec = importlib.util.spec_from_file_location("graph_xor_r1_1_runtime", R11_ROOT / "run_r1_1.py")
if _r11_spec is None or _r11_spec.loader is None:
    raise RuntimeError("could not load the immutable R1.1 runner")
r11 = importlib.util.module_from_spec(_r11_spec)
sys.modules[_r11_spec.name] = r11
_r11_spec.loader.exec_module(r11)


PLAN_PATH = HERE / "R1_2_ICL_FORMATION.md"
RESULT_PATH = HERE / "r1_2_results.json"
FIGURE_PATH = HERE / "r1_2_formation_matrix.png"
SCHEMA_VERSION = "1.0"
CASE_SEED = 202_608_171_742
SHOT_COUNTS = (4, 16, 64)
SURFACES = ("ICL-P2", "ICL-P3", "ICL-P5", "ICL-B")
CONDITIONS = ("correct", "label_shuffled")
TARGET_CASES = 24
DEMO_BANK_SIZE = 64
FORWARD_CEILING = 1_500
MODEL_CONTEXT = 40_960
MODEL_ORDER = (
    ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
)
DIRECT_LITERALS = ("0", "1")
RUNTIME_LOCK = dict(r11.RUNTIME_LOCK)
IMAGE_BASE = r11.IMAGE_BASE
MODAL_SDK_VERSION = r11.MODAL_SDK_VERSION
MODAL_APP = "graph_xor-r1-2-icl-formation"
MODAL_VOLUME = r11.MODAL_VOLUME
GPU_REQUEST = "L40S"
CAMPAIGN_SECONDS_CEILING = 7_200
GPU_MEMORY_MIN_BYTES = 40 * 1024**3

EXPECTED_V1 = {
    "spec_sha256": "fa6afd7502da3d4edd348fb626b9585344ee276cb8b3cef3d5b059e143a27e87",
    "manifest_sha256": "bf02600c469c0f129e87a47fe9e6663a91ccd741827b0aa3eef5ce535fbfda14",
    "b0_package_sha256": "a8df19e8bc1e62fd284093ba9a24df0d905717eb89366dcedacb1f27bd9e6d8d",
}
EXPECTED_R11 = {
    "plan_sha256": "579e10615bf0244d447cafa9a11179b7d011927a84cec5c42a948435bfc6d62b",
    "runner_sha256": "6034b778e50926aeedf377e8402f0e571614d6985ffef924cd89095b65454324",
    "result_sha256": "b33a9940dc24b0610aca66641eb2e3d8f292f4247ca25f809a6962dc4f38f1c0",
    "figure_sha256": "7418e0f4d54a5f87c168a497a3de4625bf7274100cf7a27bbaffe7038895f163",
}

OA1_PRE_AMENDMENT_PLAN_SHA256 = "d749b4b20a6d0d2978a82bef72e1905eecda1a316a5337f88ee4c0c87929b83b"
OA1_PRE_AMENDMENT_RUNNER_SHA256 = "1a718c8b019c055f00976fd96adbc6b326e76184c387036297d5800f948f8654"
OA1_ABORT_RESULT_SHA256 = "d0f1ce479e3343b61b54ef552473ffb7d8d08fb799922790d4f2b9405a573a78"
OA1_SEGMENT_1_RECORD_LEDGER_SHA256 = "7a0f73cc362940b2126236837526e8c47318d8f9fe1d302e8d7a73069f411914"
OA1_SEGMENT_1_RECORD_COUNT = 24
OA1_RESUME_CURSOR = {
    "model_id": "Qwen/Qwen3-4B",
    "surface": "ICL-P2",
    "shot_count": 4,
    "condition": "label_shuffled",
    "first_case_id": "3dece6af0b82f7b2420829909df7ba905ee016b839f88175c9646082a8a053f9",
}

TOKEN_BINDING: dict[str, Any] = json.loads(r'''{"acquisition":{"Qwen/Qwen3-4B":{"files":[{"name":"config.json","sha256":"8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a","size_bytes":726},{"name":"merges.txt","sha256":"8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5","size_bytes":1671853},{"name":"tokenizer.json","sha256":"aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4","size_bytes":11422654},{"name":"tokenizer_config.json","sha256":"d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101","size_bytes":9732},{"name":"vocab.json","sha256":"ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910","size_bytes":2776833}],"revision":"1cfa9a7208912126459214e8b04321603b3df60c"},"Qwen/Qwen3-8B":{"files":[{"name":"config.json","sha256":"f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30","size_bytes":728},{"name":"merges.txt","sha256":"8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5","size_bytes":1671853},{"name":"tokenizer.json","sha256":"aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4","size_bytes":11422654},{"name":"tokenizer_config.json","sha256":"d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101","size_bytes":9732},{"name":"vocab.json","sha256":"ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910","size_bytes":2776833}],"revision":"b968826d9c46dd6066d109eabc6255188de91218"}},"aggregate_sha256":"0664a1e04fb6c9c82ab953486d54b93171b08c433934fca45ceaec4bfdccf768","case_counts":{"ICL-B":144,"ICL-P2":144,"ICL-P3":144,"ICL-P5":144},"case_ledger_sha256":"8869d95aed9f88c2f8557f5fc538eab34c8a396bb48a61bbb46f52143f1d3bc9","chat_template_sha256":"a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8","demo_counts":{"ICL-B":64,"ICL-P2":64,"ICL-P3":64,"ICL-P5":64},"direct_literal_token_ids":[15,16],"maximum_prefix_token_positions":29941,"models":{"Qwen/Qwen3-4B":{"cases_audited":576,"chat_template_sha256":"a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8","context_margin_positions":11018,"correct_shuffled_exact_token_count_and_multiset_match":true,"literal_token_ids":{"0":15,"1":16},"maximum_prefix_token_positions":29941,"prefix_aggregate_sha256":"c9af47f32405cbc596d2e113d93a0f8919f8d967a485bed19e39f20e223ffc4a","revision":"1cfa9a7208912126459214e8b04321603b3df60c"},"Qwen/Qwen3-8B":{"cases_audited":576,"chat_template_sha256":"a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8","context_margin_positions":11018,"correct_shuffled_exact_token_count_and_multiset_match":true,"literal_token_ids":{"0":15,"1":16},"maximum_prefix_token_positions":29941,"prefix_aggregate_sha256":"c9af47f32405cbc596d2e113d93a0f8919f8d967a485bed19e39f20e223ffc4a","revision":"b968826d9c46dd6066d109eabc6255188de91218"}},"runtime":{"huggingface_hub":"1.27.0","jinja2":"3.1.6","tokenizers":"0.22.2","transformers":"5.15.0"},"status":"TOKENIZER_ONLY_COMPILE_PASS","target_counts":{"ICL-B":24,"ICL-P2":24,"ICL-P3":24,"ICL-P5":24}}''')


class ScoutError(RuntimeError):
    """Fail-closed R1.2 compilation or execution error."""


@dataclass(frozen=True)
class TaskItem:
    item_id: str
    surface: str
    pair_id: str
    semantic_answer: int
    body: str
    strata: Mapping[str, str]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    surface: str
    shot_count: int
    condition: str
    pair_id: str
    semantic_answer: int
    prompt: str
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
    return random.Random(_stable_u64("graph_xor-r1-2", CASE_SEED, *parts))


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def _item(
    *, surface: str, item_id: str, pair_id: str, semantic_answer: int,
    body: str, strata: Mapping[str, object], metadata: Mapping[str, Any],
) -> TaskItem:
    return TaskItem(
        item_id=item_id,
        surface=surface,
        pair_id=pair_id,
        semantic_answer=int(semantic_answer),
        body=body.rstrip(),
        strata={str(key): str(value) for key, value in strata.items()},
        metadata=dict(metadata),
    )


def _p2_body(tag: str, first: int, second: int) -> str:
    a, b, c = f"{tag}A", f"{tag}B", f"{tag}C"
    return (
        f"{a} XOR {b} = {first}.\n"
        f"{b} XOR {c} = {second}.\n"
        f"QUERY: What is {a} XOR {c}?"
    )


def _p3_body(bits: Sequence[int]) -> str:
    return (
        f"BIT SEQUENCE (length {len(bits)}): " + " ".join(map(str, bits))
        + "\nQUERY: Return the XOR of all bits."
    )


def _p5_body(sample: Any) -> str:
    return (
        "Inspect the complete graph-XOR world.\n"
        + r11.render_world(sample, "symbolic_xor")
        + "QUERY: Return the XOR parity around the unique cycle."
    )


def _theta_query(alpha: Sequence[int]) -> str:
    if tuple(alpha) == (1, 0):
        return "Return the XOR parity on the cycle formed by theta paths P0 and P2."
    if tuple(alpha) == (0, 1):
        return "Return the XOR parity on the cycle formed by theta paths P1 and P2."
    return "Return the XOR parity on the cycle formed by theta paths P0 and P1."


def _b_body(sample: Any, alpha: Sequence[int]) -> str:
    return (
        "Inspect the complete graph-XOR theta world. A theta path Pk is the s-to-t "
        "chain through nodes beginning pk:v; gadget nodes containing :g are not part "
        "of a queried cycle.\n"
        + r11.render_world(sample, "symbolic_xor")
        + "QUERY: " + _theta_query(alpha)
    )


def _bits_for_prefix(prefix: Sequence[int], answer: int) -> tuple[int, ...]:
    final = int(answer)
    for bit in prefix:
        final ^= int(bit)
    return tuple(map(int, prefix)) + (final,)


def _fresh_cycle_prefixes(count: int) -> list[tuple[int, ...]]:
    historical = {
        tuple(case.metadata["bits"][:-1])
        for case in r11.compile_cases()["P3"]
    }
    candidates = [tuple(map(int, f"{value:07b}")) for value in range(128)]
    candidates = [prefix for prefix in candidates if prefix not in historical]
    _rng("cycle-prefixes").shuffle(candidates)
    if len(candidates) < count:
        raise ScoutError("not enough fresh cycle prefixes")
    return candidates[:count]


def _cycle_sample(prefix: Sequence[int], answer: int, identity: str) -> tuple[Any, tuple[int, ...]]:
    bits = _bits_for_prefix(prefix, answer)
    world = r11.generate_unicyclic_world(
        cycle_length=8, tree_edges=0,
        seed=_stable_u64("r12-cycle-world", identity) % (2**31),
    )
    sample = r11.LabeledWorld(world, bits)
    order = list(range(8))
    _rng("cycle-order", identity).shuffle(order)
    if order == list(range(8)):
        order = order[1:] + order[:1]
    return r11.reorder_sample(sample, order), bits


def _fresh_theta_worlds(count: int) -> list[Any]:
    excluded = {
        case.metadata["world_hash"]
        for case in r11.compile_cases()["B"]
    }
    worlds: list[Any] = []
    seen = set(excluded)
    cursor = 0
    while len(worlds) < count:
        seed = _stable_u64("r12-theta-world", cursor) % (2**31)
        world = r11.generate_decorated_theta(
            path_internal_count=3, max_gadget_nodes=3, seed=seed
        )
        cursor += 1
        if world.group_hash not in seen:
            seen.add(world.group_hash)
            worlds.append(world)
        if cursor > 20_000:
            raise ScoutError("could not generate fresh decorated-theta worlds")
    return worlds


def _source_class(desired: Sequence[int], permutation: Sequence[int]) -> tuple[int, int]:
    matrix = r11.path_permutation_matrix(permutation)
    for candidate in r11.CLASSES:
        if r11.apply_matrix(matrix, candidate) == tuple(desired):
            return candidate
    raise ScoutError("theta path action has no inverse class")


def _theta_item(
    *, surface: str, item_id: str, pair_id: str, world: Any,
    alpha: Sequence[int], answer: int, selector: int,
) -> TaskItem:
    options = [candidate for candidate in r11.CLASSES if r11.query_answer(candidate, alpha) == answer]
    desired_c = options[selector % len(options)]
    permutation = r11.PATH_PERMUTATIONS[selector % len(r11.PATH_PERMUTATIONS)]
    source_c = _source_class(desired_c, permutation)
    source = r11.labeled_theta_for_class(
        world, source_c,
        gauge_seed=_stable_u64("r12-theta-gauge", item_id),
    )
    sample = r11.permute_theta_sample(source, permutation)
    actual_c = r11.class_from_path_parities(r11.theta_path_parities(sample))
    if actual_c != desired_c or r11.query_answer(actual_c, alpha) != answer:
        raise ScoutError("theta item violates alpha^T C")
    return _item(
        surface=surface,
        item_id=item_id,
        pair_id=pair_id,
        semantic_answer=answer,
        body=_b_body(sample, alpha),
        strata={"semantic_answer": answer, "alpha": f"{alpha[0]}{alpha[1]}"},
        metadata={
            "world_hash": world.group_hash,
            "C": list(desired_c),
            "alpha": list(alpha),
            "path_permutation": list(permutation),
        },
    )


def _build_p2() -> tuple[list[TaskItem], list[TaskItem]]:
    demos: list[TaskItem] = []
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for block in range(16):
        order = list(combos)
        _rng("p2-demo-order", block).shuffle(order)
        for slot, (first, second) in enumerate(order):
            answer = first ^ second
            index = block * 4 + slot
            demos.append(_item(
                surface="ICL-P2", item_id=f"P2-demo-{index:03d}",
                pair_id=f"P2-demo-{index:03d}", semantic_answer=answer,
                body=_p2_body(f"d{index:03d}", first, second),
                strata={"semantic_answer": answer},
                metadata={"edge_bits": [first, second], "role": "demo"},
            ))
    targets: list[TaskItem] = []
    for pair_index in range(12):
        first = pair_index % 2
        for answer in (0, 1):
            second = first ^ answer
            targets.append(_item(
                surface="ICL-P2", item_id=f"P2-target-{pair_index:02d}-{answer}",
                pair_id=f"P2-target-{pair_index:02d}", semantic_answer=answer,
                body=_p2_body(f"t{pair_index:02d}{answer}", first, second),
                strata={"semantic_answer": answer, "first_edge": first},
                metadata={"edge_bits": [first, second], "role": "target"},
            ))
    return demos, targets


def _build_p3_p5() -> dict[str, tuple[list[TaskItem], list[TaskItem]]]:
    prefixes = _fresh_cycle_prefixes(44)
    result: dict[str, tuple[list[TaskItem], list[TaskItem]]] = {}
    for surface in ("ICL-P3", "ICL-P5"):
        demos: list[TaskItem] = []
        targets: list[TaskItem] = []
        for pair_index, prefix in enumerate(prefixes[:32]):
            for answer in (0, 1):
                item_id = f"{surface}-demo-{pair_index:02d}-{answer}"
                if surface == "ICL-P3":
                    bits = _bits_for_prefix(prefix, answer)
                    body = _p3_body(bits)
                else:
                    sample, bits = _cycle_sample(prefix, answer, item_id)
                    body = _p5_body(sample)
                demos.append(_item(
                    surface=surface, item_id=item_id,
                    pair_id=f"{surface}-demo-{pair_index:02d}",
                    semantic_answer=answer, body=body,
                    strata={"semantic_answer": answer, "parity_length": 8},
                    metadata={"bits": list(bits), "role": "demo"},
                ))
        for local_index, prefix in enumerate(prefixes[32:44]):
            for answer in (0, 1):
                item_id = f"{surface}-target-{local_index:02d}-{answer}"
                if surface == "ICL-P3":
                    bits = _bits_for_prefix(prefix, answer)
                    body = _p3_body(bits)
                else:
                    sample, bits = _cycle_sample(prefix, answer, item_id)
                    body = _p5_body(sample)
                targets.append(_item(
                    surface=surface, item_id=item_id,
                    pair_id=f"cycle-target-{local_index:02d}",
                    semantic_answer=answer, body=body,
                    strata={"semantic_answer": answer, "parity_length": 8},
                    metadata={"bits": list(bits), "role": "target"},
                ))
        result[surface] = (demos, targets)
    return result


def _build_b() -> tuple[list[TaskItem], list[TaskItem]]:
    worlds = _fresh_theta_worlds(68)
    demos: list[TaskItem] = []
    for index, world in enumerate(worlds[:64]):
        answer = (index % 4) % 2
        alpha = r11.ALPHAS[index % len(r11.ALPHAS)]
        demos.append(_theta_item(
            surface="ICL-B", item_id=f"B-demo-{index:03d}",
            pair_id=f"B-demo-{index:03d}", world=world,
            alpha=alpha, answer=answer, selector=index,
        ))
    targets: list[TaskItem] = []
    for world_index, world in enumerate(worlds[64:]):
        for alpha_index, alpha in enumerate(r11.ALPHAS):
            for answer in (0, 1):
                targets.append(_theta_item(
                    surface="ICL-B",
                    item_id=f"B-target-{world_index}-{alpha_index}-{answer}",
                    pair_id=f"B-target-{world.group_hash}-{alpha_index}",
                    world=world, alpha=alpha, answer=answer,
                    selector=world_index + alpha_index * 2 + answer,
                ))
    return demos, targets


def compile_task_banks() -> dict[str, dict[str, list[TaskItem]]]:
    p2_demos, p2_targets = _build_p2()
    p3_p5 = _build_p3_p5()
    b_demos, b_targets = _build_b()
    banks = {
        "ICL-P2": {"demos": p2_demos, "targets": p2_targets},
        "ICL-P3": {"demos": p3_p5["ICL-P3"][0], "targets": p3_p5["ICL-P3"][1]},
        "ICL-P5": {"demos": p3_p5["ICL-P5"][0], "targets": p3_p5["ICL-P5"][1]},
        "ICL-B": {"demos": b_demos, "targets": b_targets},
    }
    _validate_task_banks(banks)
    return banks


def _shuffled_labels(demos: Sequence[TaskItem]) -> list[int]:
    labels = [item.semantic_answer for item in demos]
    shuffled: list[int] = []
    for block_start in range(0, len(labels), 4):
        block = labels[block_start:block_start + 4]
        if Counter(block) != {0: 2, 1: 2}:
            raise ScoutError("demo label block is not exactly balanced")
        candidates = []
        for permutation in itertools.permutations(range(4)):
            candidate = [block[index] for index in permutation]
            changed = sum(left != right for left, right in zip(block, candidate))
            if changed == 2 and candidate != block and candidate != [1 - value for value in block]:
                candidates.append(candidate)
        if not candidates:
            raise ScoutError("could not construct balanced shuffled control")
        shuffled.extend(candidates[_stable_u64("shuffle-block", block_start, block) % len(candidates)])
    return shuffled


def _icl_prompt(demos: Sequence[TaskItem], demo_labels: Sequence[int], target: TaskItem) -> str:
    if len(demos) != len(demo_labels):
        raise ScoutError("demo inputs and labels differ in length")
    sections = [
        "Infer the exact binary task from the input-output demonstrations below.",
        "Use only the demonstrated input-output relation. Do not produce reasoning or a scratchpad.",
    ]
    for index, (demo, label) in enumerate(zip(demos, demo_labels), start=1):
        sections.append(
            f"DEMONSTRATION {index:02d}\nINPUT:\n{demo.body}\nOUTPUT: {label}"
        )
    sections.append(
        "TARGET INPUT:\n" + target.body
        + "\nReturn the semantic bit itself. Output exactly 0 or 1 and nothing else.\nANSWER:"
    )
    return "\n\n".join(sections)


def compile_eval_cases() -> tuple[dict[str, list[EvalCase]], dict[str, dict[str, list[TaskItem]]]]:
    banks = compile_task_banks()
    result: dict[str, list[EvalCase]] = {surface: [] for surface in SURFACES}
    for surface in SURFACES:
        demos = banks[surface]["demos"]
        shuffled = _shuffled_labels(demos)
        correct = [item.semantic_answer for item in demos]
        for shot_count in SHOT_COUNTS:
            for condition, labels in (("correct", correct), ("label_shuffled", shuffled)):
                prefix = demos[:shot_count]
                label_prefix = labels[:shot_count]
                for target in banks[surface]["targets"]:
                    prompt = _icl_prompt(prefix, label_prefix, target)
                    identity = {
                        "surface": surface,
                        "shot_count": shot_count,
                        "condition": condition,
                        "target_id": target.item_id,
                        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                    }
                    result[surface].append(EvalCase(
                        case_id=_sha256_bytes(_canonical_bytes(("graph_xor-r1-2-case", identity))),
                        surface=surface,
                        shot_count=shot_count,
                        condition=condition,
                        pair_id=target.pair_id,
                        semantic_answer=target.semantic_answer,
                        prompt=prompt,
                        strata=dict(target.strata),
                        metadata={
                            "target_item_id": target.item_id,
                            "demo_item_ids": [item.item_id for item in prefix],
                            "demo_labels_sha256": _sha256_bytes(_canonical_bytes(label_prefix)),
                            "target_metadata": dict(target.metadata),
                        },
                    ))
    _validate_eval_cases(result, banks)
    return result, banks


def _validate_task_banks(banks: Mapping[str, Mapping[str, Sequence[TaskItem]]]) -> None:
    for surface in SURFACES:
        demos = list(banks[surface]["demos"])
        targets = list(banks[surface]["targets"])
        if len(demos) != DEMO_BANK_SIZE or len(targets) != TARGET_CASES:
            raise ScoutError(f"{surface} bank has wrong size")
        if len({item.item_id for item in demos + targets}) != len(demos) + len(targets):
            raise ScoutError(f"{surface} item IDs are not unique")
        if set(item.body for item in demos) & set(item.body for item in targets):
            raise ScoutError(f"{surface} demo and target bodies overlap")
        for prefix in SHOT_COUNTS:
            if Counter(item.semantic_answer for item in demos[:prefix]) != {0: prefix // 2, 1: prefix // 2}:
                raise ScoutError(f"{surface} {prefix}-shot prefix is not balanced")
        if Counter(item.semantic_answer for item in targets) != {0: 12, 1: 12}:
            raise ScoutError(f"{surface} target labels are not balanced")
        pairs: dict[str, list[int]] = defaultdict(list)
        for item in targets:
            pairs[item.pair_id].append(item.semantic_answer)
        if len(pairs) != 12 or any(sorted(values) != [0, 1] for values in pairs.values()):
            raise ScoutError(f"{surface} targets are not 12 exact matched pairs")
        shuffled = _shuffled_labels(demos)
        for prefix in SHOT_COUNTS:
            original = [item.semantic_answer for item in demos[:prefix]]
            control = shuffled[:prefix]
            if Counter(control) != Counter(original):
                raise ScoutError(f"{surface} shuffled label count drift")
            if sum(a != b for a, b in zip(original, control)) != prefix // 2:
                raise ScoutError(f"{surface} shuffled corruption is not exactly 50%")
    p3_demo = {tuple(item.metadata["bits"]) for item in banks["ICL-P3"]["demos"]}
    p3_target = {tuple(item.metadata["bits"]) for item in banks["ICL-P3"]["targets"]}
    if p3_demo & p3_target or len(p3_demo) != 64 or len(p3_target) != 24:
        raise ScoutError("P3 demo/target bit patterns are not disjoint and unique")
    for role in ("demos", "targets"):
        p3 = [tuple(item.metadata["bits"]) for item in banks["ICL-P3"][role]]
        p5 = [tuple(item.metadata["bits"]) for item in banks["ICL-P5"][role]]
        if p3 != p5:
            raise ScoutError("P3/P5 lost their shared underlying bit ledger")
    b_demo_worlds = {item.metadata["world_hash"] for item in banks["ICL-B"]["demos"]}
    b_target_worlds = {item.metadata["world_hash"] for item in banks["ICL-B"]["targets"]}
    if len(b_demo_worlds) != 64 or len(b_target_worlds) != 4 or b_demo_worlds & b_target_worlds:
        raise ScoutError("B demo and target world morphologies are not disjoint")
    b_combos = Counter(
        (tuple(item.metadata["C"]), tuple(item.metadata["alpha"]))
        for item in banks["ICL-B"]["targets"]
    )
    if set(b_combos) != set(itertools.product(r11.CLASSES, r11.ALPHAS)) or set(b_combos.values()) != {2}:
        raise ScoutError("B target C x alpha schedule is not exactly balanced")


def _validate_eval_cases(
    cases: Mapping[str, Sequence[EvalCase]],
    banks: Mapping[str, Mapping[str, Sequence[TaskItem]]],
) -> None:
    all_ids: set[str] = set()
    for surface in SURFACES:
        rows = list(cases[surface])
        if len(rows) != len(SHOT_COUNTS) * len(CONDITIONS) * TARGET_CASES:
            raise ScoutError(f"{surface} eval case count is wrong")
        for shot_count in SHOT_COUNTS:
            for condition in CONDITIONS:
                cell = [row for row in rows if row.shot_count == shot_count and row.condition == condition]
                if len(cell) != TARGET_CASES:
                    raise ScoutError(f"{surface} {shot_count} {condition} cell size is wrong")
                if Counter(row.semantic_answer for row in cell) != {0: 12, 1: 12}:
                    raise ScoutError(f"{surface} target cell is not balanced")
        if any(row.case_id in all_ids for row in rows):
            raise ScoutError("eval case ID collision")
        all_ids.update(row.case_id for row in rows)
        targets = {item.item_id for item in banks[surface]["targets"]}
        if {row.metadata["target_item_id"] for row in rows} != targets:
            raise ScoutError(f"{surface} target ledger differs across conditions")
    if 2 * sum(len(rows) for rows in cases.values()) > FORWARD_CEILING:
        raise ScoutError("worst-case scientific forward count exceeds ceiling")


def _public_item(item: TaskItem) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "surface": item.surface,
        "pair_id": item.pair_id,
        "semantic_answer": item.semantic_answer,
        "body_sha256": _sha256_bytes(item.body.encode("utf-8")),
        "strata": dict(item.strata),
        "metadata": dict(item.metadata),
    }


def _public_case(case: EvalCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "surface": case.surface,
        "shot_count": case.shot_count,
        "condition": case.condition,
        "pair_id": case.pair_id,
        "semantic_answer": case.semantic_answer,
        "prompt_sha256": _sha256_bytes(case.prompt.encode("utf-8")),
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
            allow_patterns=list(r11.TOKENIZER_FILES),
            cache_dir=str(cache_root),
        ))
        files = sorted(path for path in snapshot.rglob("*") if path.is_file())
        names = {path.relative_to(snapshot).as_posix() for path in files}
        if names != set(r11.TOKENIZER_FILES):
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


def _case_ledger(
    cases: Mapping[str, Sequence[EvalCase]],
    banks: Mapping[str, Mapping[str, Sequence[TaskItem]]],
) -> dict[str, Any]:
    return {
        "banks": {
            surface: {
                role: [_public_item(item) for item in banks[surface][role]]
                for role in ("demos", "targets")
            }
            for surface in SURFACES
        },
        "eval_cases": {
            surface: [_public_case(case) for case in cases[surface]]
            for surface in SURFACES
        },
    }


def tokenizer_only_compile(cache_root: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    cases, banks = compile_eval_cases()
    acquisition, snapshots = _download_tokenizers(cache_root)
    ledger_sha = _sha256_bytes(_canonical_bytes(_case_ledger(cases, banks)))
    model_results: dict[str, Any] = {}
    for model_id, revision in MODEL_ORDER:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshots[model_id], use_fast=True, local_files_only=True, trust_remote_code=False
        )
        literal_ids: dict[str, set[int]] = defaultdict(set)
        prefix_records: list[dict[str, Any]] = []
        token_records: dict[tuple[str, int, str], list[int]] = {}
        maximum_positions = 0
        for surface in SURFACES:
            for case in cases[surface]:
                text, ids = r11._chat_prefix(tokenizer, case.prompt)
                maximum_positions = max(maximum_positions, len(ids))
                if len(ids) + 1 > MODEL_CONTEXT:
                    raise ScoutError(
                        f"{model_id} {surface} {case.shot_count}-shot exceeds context: {len(ids)}"
                    )
                for literal in DIRECT_LITERALS:
                    literal_ids[literal].add(r11._appended_token_id(tokenizer, text, ids, literal))
                prefix_records.append({
                    "case_id": case.case_id,
                    "token_ids_sha256": _sha256_bytes(_canonical_bytes(ids)),
                    "token_positions": len(ids),
                })
                key = (surface, case.shot_count, case.metadata["target_item_id"])
                token_records[(key[0], key[1], key[2] + ":" + case.condition)] = ids
        for surface in SURFACES:
            for shot_count in SHOT_COUNTS:
                for target in banks[surface]["targets"]:
                    correct = token_records[(surface, shot_count, target.item_id + ":correct")]
                    shuffled = token_records[(surface, shot_count, target.item_id + ":label_shuffled")]
                    if len(correct) != len(shuffled) or Counter(correct) != Counter(shuffled):
                        raise ScoutError(
                            f"{model_id} correct/shuffled token matching failed for "
                            f"{surface} {shot_count} {target.item_id}"
                        )
        if any(len(values) != 1 for values in literal_ids.values()):
            raise ScoutError(f"{model_id} has context-dependent direct token IDs")
        model_results[model_id] = {
            "revision": revision,
            "chat_template_sha256": _sha256_bytes(str(tokenizer.chat_template).encode("utf-8")),
            "literal_token_ids": {
                literal: next(iter(values)) for literal, values in sorted(literal_ids.items())
            },
            "prefix_aggregate_sha256": _sha256_bytes(_canonical_bytes(prefix_records)),
            "maximum_prefix_token_positions": maximum_positions,
            "cases_audited": len(prefix_records),
            "correct_shuffled_exact_token_count_and_multiset_match": True,
            "context_margin_positions": MODEL_CONTEXT - maximum_positions - 1,
        }
    direct_sets = {
        tuple(result["literal_token_ids"][literal] for literal in DIRECT_LITERALS)
        for result in model_results.values()
    }
    template_sets = {result["chat_template_sha256"] for result in model_results.values()}
    if len(direct_sets) != 1 or len(template_sets) != 1:
        raise ScoutError("model-family direct tokens or chat template differ")
    aggregate = {
        "status": "TOKENIZER_ONLY_COMPILE_PASS",
        "models": model_results,
        "acquisition": acquisition,
        "case_ledger_sha256": ledger_sha,
        "case_counts": {surface: len(cases[surface]) for surface in SURFACES},
        "demo_counts": {surface: len(banks[surface]["demos"]) for surface in SURFACES},
        "target_counts": {surface: len(banks[surface]["targets"]) for surface in SURFACES},
        "direct_literal_token_ids": list(next(iter(direct_sets))),
        "chat_template_sha256": next(iter(template_sets)),
        "maximum_prefix_token_positions": max(
            result["maximum_prefix_token_positions"] for result in model_results.values()
        ),
        "runtime": {
            key: importlib.metadata.version(package)
            for key, package in (
                ("transformers", "transformers"),
                ("tokenizers", "tokenizers"),
                ("huggingface_hub", "huggingface-hub"),
                ("jinja2", "jinja2"),
            )
        },
    }
    aggregate["aggregate_sha256"] = _sha256_bytes(_canonical_bytes(aggregate))
    return aggregate


def _cell_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return r11.classify_surface(records)


def _formation_assessment(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    passing: list[int] = []
    signal: list[int] = []
    contrasts: dict[str, dict[str, float | bool | str]] = {}
    for shot_count in SHOT_COUNTS:
        correct = cells[str(shot_count)]["correct"]["metrics"]
        shuffled = cells[str(shot_count)]["label_shuffled"]["metrics"]
        accuracy_delta = correct["semantic_accuracy"] - shuffled["semantic_accuracy"]
        pair_delta = (
            correct["paired_directional_consistency"]
            - shuffled["paired_directional_consistency"]
        )
        auc_delta = correct["semantic_score_auc"] - shuffled["semantic_score_auc"]
        passed = (
            correct["classification"] == "BEHAVIOR_PASS"
            and shuffled["classification"] != "BEHAVIOR_PASS"
            and accuracy_delta >= 0.125
            and pair_delta >= 0.125
        )
        signaled = (
            correct["classification"] == "SCORE_SIGNAL_ONLY"
            and (auc_delta >= 0.125 or pair_delta >= 0.125)
        )
        if passed:
            passing.append(shot_count)
        if signaled:
            signal.append(shot_count)
        contrasts[str(shot_count)] = {
            "correct_classification": correct["classification"],
            "label_shuffled_classification": shuffled["classification"],
            "accuracy_delta": accuracy_delta,
            "paired_consistency_delta": pair_delta,
            "auc_delta": auc_delta,
            "formation_pass": passed,
            "formation_signal_only": signaled,
        }
    if passing:
        outcome = "FORMATION_PASS"
        selected = min(passing)
    elif signal:
        outcome = "FORMATION_SIGNAL_ONLY"
        selected = None
    else:
        outcome = "NO_DETECTED_FORMATION"
        selected = None
    return {
        "outcome": outcome,
        "selected_shot_count": selected,
        "passing_shot_counts": passing,
        "signal_only_shot_counts": signal,
        "contrasts": contrasts,
    }


def _verify_immutable_ancestors() -> dict[str, Any]:
    actual_v1 = {
        "spec_sha256": _sha256_file(V1_ROOT / "GRAPH_XOR_R1_GLOBAL_OBSTRUCTION_SPEC_v1.0.md"),
        "manifest_sha256": _sha256_file(V1_ROOT / "study_manifest.json"),
        "b0_package_sha256": r11.package_sha256(),
    }
    if actual_v1 != EXPECTED_V1:
        raise ScoutError(f"immutable R1 v1.0 binding drift: {actual_v1}")
    manifest = json.loads((V1_ROOT / "study_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"]["current_stage"] != "CLOSED_AT_C0" or manifest["status"]["s0"] != "S0_SELECTION_FAIL":
        raise ScoutError("R1 v1.0 closeout state drifted")
    actual_r11 = {
        "plan_sha256": _sha256_file(R11_ROOT / "R1_1_CAPABILITY_LOCALIZATION.md"),
        "runner_sha256": _sha256_file(R11_ROOT / "run_r1_1.py"),
        "result_sha256": _sha256_file(R11_ROOT / "r1_1_results.json"),
        "figure_sha256": _sha256_file(R11_ROOT / "r1_1_capability_matrix.png"),
    }
    if actual_r11 != EXPECTED_R11:
        raise ScoutError(f"immutable R1.1 binding drift: {actual_r11}")
    historical = json.loads((R11_ROOT / "r1_1_results.json").read_text(encoding="utf-8"))
    if historical["status"] != "SCOUT_CLOSE_NO_DIRECT_B_THROUGH_8B":
        raise ScoutError("R1.1 is not in its frozen closeout state")
    return {"r1_v1": actual_v1, "r1_1": actual_r11}


def _historical_zero_shot_reference() -> dict[str, Any]:
    historical = json.loads((R11_ROOT / "r1_1_results.json").read_text(encoding="utf-8"))
    mapping = {"ICL-P2": "P2", "ICL-P3": "P3", "ICL-P5": "P5", "ICL-B": "B"}
    result: dict[str, Any] = {}
    for model_id, _ in MODEL_ORDER:
        result[model_id] = {}
        for current, old in mapping.items():
            entry = historical["models"][model_id]["surfaces"].get(old)
            result[model_id][current] = (
                {"status": "HISTORICAL_OPENED", "metrics": entry["metrics"]}
                if entry else {"status": "HISTORICAL_UNOPENED"}
            )
    return result


def _runtime_environment() -> dict[str, Any]:
    return r11._runtime_environment()


def _verify_runtime() -> dict[str, Any]:
    import torch

    runtime = _runtime_environment()
    for key, expected in RUNTIME_LOCK.items():
        if runtime.get(key) != expected:
            raise ScoutError(f"runtime drift for {key}: {runtime.get(key)!r} != {expected!r}")
    if not runtime["cuda_available"] or "L40S" not in str(runtime["cuda_device_name"]).upper():
        raise ScoutError(f"campaign did not receive frozen L40S: {runtime['cuda_device_name']}")
    if int(runtime["cuda_device_total_memory_bytes"]) < GPU_MEMORY_MIN_BYTES:
        raise ScoutError("campaign GPU has less than frozen 40 GiB minimum")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(CASE_SEED)
    torch.cuda.manual_seed_all(CASE_SEED)
    return runtime


def _verify_token_binding(
    model_id: str, tokenizer: Any, cases: Mapping[str, Sequence[EvalCase]]
) -> None:
    expected = TOKEN_BINDING["models"][model_id]
    if _sha256_bytes(str(tokenizer.chat_template).encode("utf-8")) != expected["chat_template_sha256"]:
        raise ScoutError(f"{model_id} chat template drift")
    literal_ids: dict[str, set[int]] = defaultdict(set)
    prefix_records: list[dict[str, Any]] = []
    for surface in SURFACES:
        for case in cases[surface]:
            text, ids = r11._chat_prefix(tokenizer, case.prompt)
            for literal in DIRECT_LITERALS:
                literal_ids[literal].add(r11._appended_token_id(tokenizer, text, ids, literal))
            prefix_records.append({
                "case_id": case.case_id,
                "token_ids_sha256": _sha256_bytes(_canonical_bytes(ids)),
                "token_positions": len(ids),
            })
    actual_ids = {
        literal: next(iter(values))
        for literal, values in literal_ids.items()
        if len(values) == 1
    }
    if actual_ids != expected["literal_token_ids"]:
        raise ScoutError(f"{model_id} direct token binding drift")
    if _sha256_bytes(_canonical_bytes(prefix_records)) != expected["prefix_aggregate_sha256"]:
        raise ScoutError(f"{model_id} prompt-prefix ledger drift")


def _score_case(model: Any, tokenizer: Any, case: EvalCase, device: str) -> dict[str, Any]:
    import torch

    text, ids = r11._chat_prefix(tokenizer, case.prompt)
    zero_id = r11._appended_token_id(tokenizer, text, ids, "0")
    one_id = r11._appended_token_id(tokenizer, text, ids, "1")
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            input_ids=tensor,
            use_cache=False,
            logits_to_keep=1,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        logits = outputs.logits[0, -1, [zero_id, one_id]].float().cpu().tolist()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    del outputs, tensor
    zero_logit, one_logit = map(float, logits)
    score = one_logit - zero_logit
    if score > 0:
        prediction, chosen_id, chosen_literal = 1, one_id, "1"
    elif score < 0:
        prediction, chosen_id, chosen_literal = 0, zero_id, "0"
    elif zero_id <= one_id:
        prediction, chosen_id, chosen_literal = 0, zero_id, "0"
    else:
        prediction, chosen_id, chosen_literal = 1, one_id, "1"
    return {
        "case_id": case.case_id,
        "surface": case.surface,
        "shot_count": case.shot_count,
        "condition": case.condition,
        "pair_id": case.pair_id,
        "semantic_answer": case.semantic_answer,
        "semantic_score": score,
        "semantic_prediction": prediction,
        "zero_logit": zero_logit,
        "one_logit": one_logit,
        "zero_token_id": zero_id,
        "one_token_id": one_id,
        "physical_choice_token_id": chosen_id,
        "physical_choice_literal": chosen_literal,
        "candidate_pair_id": "direct-01",
        "entry_position": None,
        "strata": dict(case.strata),
        "prompt_sha256": _sha256_bytes(case.prompt.encode("utf-8")),
        "prefix_token_ids_sha256": _sha256_bytes(_canonical_bytes(ids)),
        "token_positions": len(ids),
        "forward_seconds": elapsed,
    }


def _new_result(
    plan_sha: str, runner_sha: str, runtime: Mapping[str, Any],
    ancestors: Mapping[str, Any], ledger_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "program_id": "GRAPH_XOR_R1_2_ICL_FORMATION",
        "status": "SCOUT_IN_PROGRESS",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan_sha256": plan_sha,
        "runner_sha256": runner_sha,
        "case_generation_seed": CASE_SEED,
        "case_ledger_sha256": ledger_sha,
        "forward_ceiling": FORWARD_CEILING,
        "forwards_completed": 0,
        "runtime_environment": dict(runtime),
        "immutable_ancestor_bindings": dict(ancestors),
        "historical_zero_shot_reference": _historical_zero_shot_reference(),
        "token_binding": TOKEN_BINDING,
        "model_order": [model_id for model_id, _ in MODEL_ORDER],
        "models": {
            model_id: {
                "revision": revision,
                "status": "UNOPENED",
                "surfaces": {},
                "failure_surface": None,
            }
            for model_id, revision in MODEL_ORDER
        },
        "stop_reason": None,
    }


def _render_figure(result: Mapping[str, Any], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rows = [(model_id, surface) for model_id, _ in MODEL_ORDER for surface in SURFACES]
    columns = [
        ("0*", None, None),
        ("4C", 4, "correct"), ("4S", 4, "label_shuffled"),
        ("16C", 16, "correct"), ("16S", 16, "label_shuffled"),
        ("64C", 64, "correct"), ("64S", 64, "label_shuffled"),
        ("OUT", None, "outcome"),
    ]
    left, top, cell_w, cell_h = 255, 96, 82, 48
    width = left + cell_w * len(columns) + 22
    height = top + cell_h * len(rows) + 82
    image = Image.new("RGB", (width, height), "#111318")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    class_colors = {
        "BEHAVIOR_PASS": "#35c46a",
        "SCORE_SIGNAL_ONLY": "#e5a93d",
        "NO_DETECTED_SIGNAL": "#d45454",
        "FORMATION_PASS": "#1ea95a",
        "FORMATION_SIGNAL_ONLY": "#d99224",
        "NO_DETECTED_FORMATION": "#bf4149",
        "UNOPENED": "#414752",
    }
    labels = {
        "BEHAVIOR_PASS": "PASS", "SCORE_SIGNAL_ONLY": "SIGNAL",
        "NO_DETECTED_SIGNAL": "NONE", "FORMATION_PASS": "FORM",
        "FORMATION_SIGNAL_ONLY": "SIGNAL", "NO_DETECTED_FORMATION": "NONE",
        "UNOPENED": "-",
    }
    draw.text((18, 18), "GRAPH_XOR R1.2 - In-Context XOR Formation", fill="white", font=font)
    draw.text(
        (18, 40),
        f"status: {result.get('status')} | forwards: {result.get('forwards_completed')}",
        fill="#b7bdc8", font=font,
    )
    draw.text((18, 62), "0* = immutable R1.1 historical zero-shot; C = correct demos; S = shuffled labels", fill="#8f98a8", font=font)
    for column, (name, _, _) in enumerate(columns):
        x = left + column * cell_w
        draw.text((x + 10, top - 25), name, fill="#dce2ed", font=font)
    for row, (model_id, surface) in enumerate(rows):
        y = top + row * cell_h
        short_model = model_id.rsplit("-", 1)[-1]
        draw.text((18, y + 15), f"{short_model}  {surface}", fill="#dce2ed", font=font)
        model_entry = result.get("models", {}).get(model_id, {})
        surface_entry = model_entry.get("surfaces", {}).get(surface, {})
        for column, (_, shot_count, condition) in enumerate(columns):
            x = left + column * cell_w
            if column == 0:
                historical = result.get("historical_zero_shot_reference", {}).get(model_id, {}).get(surface, {})
                classification = historical.get("metrics", {}).get("classification", "UNOPENED")
                value = historical.get("metrics", {}).get("semantic_accuracy")
            elif condition == "outcome":
                assessment = surface_entry.get("assessment") or {}
                classification = assessment.get("outcome", "UNOPENED")
                value = None
            else:
                cell = surface_entry.get("cells", {}).get(str(shot_count), {}).get(condition, {})
                classification = cell.get("metrics", {}).get("classification", "UNOPENED")
                value = cell.get("metrics", {}).get("semantic_accuracy")
            draw.rectangle(
                (x, y, x + cell_w - 6, y + cell_h - 6),
                fill=class_colors[classification], outline="#858b95",
            )
            label = labels[classification]
            if value is not None and classification != "UNOPENED":
                label = f"{label} {int(round(100 * float(value)))}"
            draw.text((x + 7, y + 15), label, fill="white", font=font)
    legend_y = top + cell_h * len(rows) + 18
    draw.text(
        (18, legend_y),
        "green pass/formation | amber score signal only | red no detected signal/formation | gray unopened",
        fill="#b7bdc8", font=font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def execute_scout(
    *, plan_path: Path, result_path: Path, figure_path: Path,
    expected_plan_sha: str, expected_runner_sha: str, cache_root: Path,
) -> dict[str, Any]:
    actual_plan_sha = _sha256_file(plan_path)
    actual_runner_sha = _sha256_file(Path(__file__))
    if actual_plan_sha != expected_plan_sha or actual_runner_sha != expected_runner_sha:
        raise ScoutError("plan or runner SHA differs from authorized binding")
    if TOKEN_BINDING.get("status") != "TOKENIZER_ONLY_COMPILE_PASS":
        raise ScoutError("runner lacks frozen tokenizer binding")
    runtime = _verify_runtime()
    ancestors = _verify_immutable_ancestors()
    cases, banks = compile_eval_cases()
    ledger_sha = _sha256_bytes(_canonical_bytes(_case_ledger(cases, banks)))
    if ledger_sha != TOKEN_BINDING["case_ledger_sha256"]:
        raise ScoutError("case ledger differs from tokenizer-only compile")
    result = _new_result(
        actual_plan_sha, actual_runner_sha, runtime, ancestors, ledger_sha
    )
    _atomic_json(result_path, result)
    _render_figure(result, figure_path)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        for model_id, revision in MODEL_ORDER:
            model_entry = result["models"][model_id]
            model_entry["status"] = "ACQUIRING"
            _atomic_json(result_path, result)
            print(f"R1.2 opening {model_id}: acquiring exact revision", flush=True)
            snapshot, inventory = r11._acquire_model(model_id, revision, cache_root)
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot, use_fast=True, local_files_only=True, trust_remote_code=False
            )
            _verify_token_binding(model_id, tokenizer, cases)
            model_entry["snapshot_inventory"] = inventory
            model_entry["status"] = "LOADING"
            _atomic_json(result_path, result)
            print(f"R1.2 opening {model_id}: loading BF16 SDPA on cuda:0", flush=True)
            load_started = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                snapshot,
                dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=False,
                attn_implementation="sdpa",
            )
            model.eval()
            torch.cuda.synchronize()
            model_entry["model_load_seconds"] = time.perf_counter() - load_started
            model_entry["status"] = "RUNNING"
            passed_b = False
            for surface in SURFACES:
                surface_entry: dict[str, Any] = {
                    "status": "OPENED_IN_PROGRESS", "cells": {}, "assessment": None,
                }
                model_entry["surfaces"][surface] = surface_entry
                for shot_count in SHOT_COUNTS:
                    surface_entry["cells"][str(shot_count)] = {}
                    for condition in CONDITIONS:
                        cell_cases = [
                            case for case in cases[surface]
                            if case.shot_count == shot_count and case.condition == condition
                        ]
                        records = [
                            _score_case(model, tokenizer, case, "cuda:0")
                            for case in cell_cases
                        ]
                        result["forwards_completed"] += len(records)
                        if result["forwards_completed"] > FORWARD_CEILING:
                            raise ScoutError("forward ceiling exceeded")
                        metrics = _cell_metrics(records)
                        surface_entry["cells"][str(shot_count)][condition] = {
                            "status": "OPENED_COMPLETE",
                            "cases": len(records),
                            "records": records,
                            "metrics": metrics,
                        }
                        print(
                            f"R1.2 {model_id} {surface} {shot_count} {condition}: "
                            f"{metrics['classification']}",
                            flush=True,
                        )
                        _atomic_json(result_path, result)
                        _render_figure(result, figure_path)
                surface_entry["assessment"] = _formation_assessment(surface_entry["cells"])
                surface_entry["status"] = "OPENED_COMPLETE"
                outcome = surface_entry["assessment"]["outcome"]
                print(f"R1.2 {model_id} {surface}: {outcome}", flush=True)
                _atomic_json(result_path, result)
                _render_figure(result, figure_path)
                if outcome != "FORMATION_PASS":
                    model_entry["failure_surface"] = surface
                    break
                if surface == "ICL-B":
                    passed_b = True
            model_entry["status"] = "COMPLETE"
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            _atomic_json(result_path, result)
            _render_figure(result, figure_path)
            if passed_b:
                result["status"] = "SCOUT_PASS_ICL_B_FORMATION"
                result["stop_reason"] = (
                    f"first ICL-B FORMATION_PASS at {model_id}; larger models remain unopened"
                )
                break
        else:
            result["status"] = "SCOUT_CLOSE_NO_ICL_XOR_FORMATION_THROUGH_8B"
            result["stop_reason"] = (
                "Qwen3-4B/8B bounded input-output-only ICL XOR-formation path closed"
            )
        result["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        result["unopened_models"] = [
            model_id for model_id, _ in MODEL_ORDER
            if result["models"][model_id]["status"] == "UNOPENED"
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


def _oa1_patch_info(pre_amendment_runner: Path) -> dict[str, Any]:
    if _sha256_file(pre_amendment_runner) != OA1_PRE_AMENDMENT_RUNNER_SHA256:
        raise ScoutError("OA-1 pre-amendment runner hash mismatch")
    old_text = pre_amendment_runner.read_text(encoding="utf-8")
    new_text = Path(__file__).read_text(encoding="utf-8")
    unified = "".join(difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile="pre_amendment_run_r1_2.py",
        tofile="oa1_amended_run_r1_2.py",
    ))
    return {
        "patch_diff_sha256": _sha256_bytes(unified.encode("utf-8")),
        "patch_unified_diff": unified,
        "pre_amendment_runner_sha256": OA1_PRE_AMENDMENT_RUNNER_SHA256,
        "amended_runner_sha256": _sha256_file(Path(__file__)),
    }


def _validate_oa1_resume_source(
    abort_result_path: Path,
    cases: Mapping[str, Sequence[EvalCase]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    abort_bytes = abort_result_path.read_bytes()
    abort_sha = _sha256_bytes(abort_bytes)
    if abort_sha != OA1_ABORT_RESULT_SHA256:
        raise ScoutError(f"OA-1 abort result hash mismatch: {abort_sha}")
    result = json.loads(abort_bytes.decode("utf-8"))
    if result.get("status") != "OPERATIONAL_ABORT":
        raise ScoutError("OA-1 source is not the frozen operational abort")
    if result.get("stop_reason") != "AttributeError: 'NoneType' object has no attribute 'get'":
        raise ScoutError("OA-1 source abort reason drifted")
    if result.get("plan_sha256") != OA1_PRE_AMENDMENT_PLAN_SHA256:
        raise ScoutError("OA-1 source plan binding drifted")
    if result.get("runner_sha256") != OA1_PRE_AMENDMENT_RUNNER_SHA256:
        raise ScoutError("OA-1 source runner binding drifted")
    if result.get("case_ledger_sha256") != TOKEN_BINDING["case_ledger_sha256"]:
        raise ScoutError("OA-1 source case ledger drifted")
    if result.get("forwards_completed") != OA1_SEGMENT_1_RECORD_COUNT:
        raise ScoutError("OA-1 source does not contain exactly 24 forwards")

    expected_cases = [
        case for case in cases["ICL-P2"]
        if case.shot_count == 4 and case.condition == "correct"
    ]
    next_cases = [
        case for case in cases["ICL-P2"]
        if case.shot_count == 4 and case.condition == "label_shuffled"
    ]
    if len(expected_cases) != 24 or not next_cases:
        raise ScoutError("OA-1 compiler no longer produces the frozen cell/cursor")
    if next_cases[0].case_id != OA1_RESUME_CURSOR["first_case_id"]:
        raise ScoutError("OA-1 resume cursor case ID drifted")

    models = result.get("models", {})
    opened_cells: list[tuple[str, str, str, str]] = []
    for model_id, model in models.items():
        for surface, surface_entry in model.get("surfaces", {}).items():
            for shot_count, shot_entry in surface_entry.get("cells", {}).items():
                for condition in shot_entry:
                    opened_cells.append((model_id, surface, shot_count, condition))
    expected_opened = [("Qwen/Qwen3-4B", "ICL-P2", "4", "correct")]
    if opened_cells != expected_opened:
        raise ScoutError(f"OA-1 source has unexpected opened cells: {opened_cells}")

    cell = models["Qwen/Qwen3-4B"]["surfaces"]["ICL-P2"]["cells"]["4"]["correct"]
    records = cell.get("records", [])
    expected_ids = [case.case_id for case in expected_cases]
    observed_ids = [record.get("case_id") for record in records]
    if observed_ids != expected_ids:
        raise ScoutError("OA-1 observed case IDs differ from exact frozen order")
    if len(observed_ids) != len(set(observed_ids)):
        raise ScoutError("OA-1 source has duplicate case IDs")
    next_ids = {case.case_id for case in next_cases}
    if set(observed_ids) & next_ids:
        raise ScoutError("OA-1 completed and resume case IDs overlap")
    expected_by_id = {case.case_id: case for case in expected_cases}
    required = {
        "case_id", "surface", "shot_count", "condition", "pair_id",
        "semantic_answer", "semantic_score", "semantic_prediction",
        "zero_logit", "one_logit", "prompt_sha256",
        "prefix_token_ids_sha256", "token_positions", "forward_seconds",
    }
    for record in records:
        if not required.issubset(record):
            raise ScoutError("OA-1 source record is incomplete")
        if record.get("error") is not None:
            raise ScoutError("OA-1 source contains a model-forward error")
        expected = expected_by_id[record["case_id"]]
        if record["prompt_sha256"] != _sha256_bytes(expected.prompt.encode("utf-8")):
            raise ScoutError("OA-1 source prompt hash drifted")
    record_sha = _sha256_bytes(_canonical_bytes(records))
    if record_sha != OA1_SEGMENT_1_RECORD_LEDGER_SHA256:
        raise ScoutError("OA-1 segment-1 record ledger hash mismatch")
    if _cell_metrics(records) != cell.get("metrics"):
        raise ScoutError("OA-1 source metrics do not recompute exactly")
    return result, {
        "abort_result_sha256": abort_sha,
        "completed_record_count": len(records),
        "completed_record_ledger_sha256": record_sha,
        "expected_case_ids": expected_ids,
        "missing_case_ids": [],
        "duplicate_case_ids": [],
        "unexpected_case_ids": [],
        "frozen_resume_cursor": dict(OA1_RESUME_CURSOR),
        "metric_recompute_match": True,
    }


def resume_scout_oa1(
    *, plan_path: Path, abort_result_path: Path, pre_amendment_runner: Path,
    result_path: Path, figure_path: Path, expected_plan_sha: str,
    expected_runner_sha: str, cache_root: Path,
) -> dict[str, Any]:
    actual_plan_sha = _sha256_file(plan_path)
    actual_runner_sha = _sha256_file(Path(__file__))
    if actual_plan_sha != OA1_PRE_AMENDMENT_PLAN_SHA256 or actual_plan_sha != expected_plan_sha:
        raise ScoutError("OA-1 scientific plan hash changed")
    if actual_runner_sha != expected_runner_sha:
        raise ScoutError("OA-1 amended runner hash differs from authorization")
    if TOKEN_BINDING.get("status") != "TOKENIZER_ONLY_COMPILE_PASS":
        raise ScoutError("OA-1 runner lacks frozen tokenizer binding")
    cases, banks = compile_eval_cases()
    ledger_sha = _sha256_bytes(_canonical_bytes(_case_ledger(cases, banks)))
    if ledger_sha != TOKEN_BINDING["case_ledger_sha256"]:
        raise ScoutError("OA-1 scientific case ledger changed")
    result, source_validation = _validate_oa1_resume_source(abort_result_path, cases)
    patch_info = _oa1_patch_info(pre_amendment_runner)
    runtime = _verify_runtime()
    ancestors = _verify_immutable_ancestors()
    if result["runtime_environment"] != runtime:
        raise ScoutError("OA-1 execution surface differs from segment 1")
    if result["immutable_ancestor_bindings"] != ancestors:
        raise ScoutError("OA-1 ancestor binding differs from segment 1")

    abort_snapshot = {
        "abort_reason": result["stop_reason"],
        "abort_timestamp": result.get("completed_at_utc"),
        "completed_record_count": source_validation["completed_record_count"],
        "completed_record_ledger_sha256": source_validation["completed_record_ledger_sha256"],
        "pre_amendment_plan_sha256": result["plan_sha256"],
        "pre_amendment_runner_sha256": result["runner_sha256"],
        "abort_result_sha256": source_validation["abort_result_sha256"],
        "frozen_resume_cursor": source_validation["frozen_resume_cursor"],
        "source_validation": {
            "expected_case_count": 24,
            "observed_case_count": 24,
            "missing_case_ids": [],
            "duplicate_case_ids": [],
            "unexpected_case_ids": [],
            "model_forward_errors": 0,
            "record_parse_errors": 0,
            "metric_recompute_match": True,
        },
    }
    result["operational_abort"] = abort_snapshot
    result["oa1_exact_resume"] = {
        "authorization": "EXACT_RESUME_ONLY",
        "authorization_basis": "VALUE_INDEPENDENT_OPERATIONAL_FAILURE",
        "allowed_patch_scope": [
            "null-safe rendering of absent/null assessment",
            "UTF-8-safe Windows console/log output",
            "fail-closed exact-resume validation and provenance recording",
        ],
        "forbidden_scientific_changes_verified": True,
        "scientific_plan_sha256_unchanged": actual_plan_sha,
        "case_ledger_sha256_unchanged": ledger_sha,
        "tokenizer_binding_sha256_unchanged": TOKEN_BINDING["aggregate_sha256"],
        "runtime_unchanged": True,
        "scientific_design_unchanged": True,
        "existing_records_reexecuted": False,
        "partial_metrics_used_for_resume_decision": False,
        "oa2_model_forward_resume_authorized": False,
        **patch_info,
    }
    result["execution_segments"] = [
        {
            "segment_id": 1,
            "status": "OPERATIONAL_ABORT",
            "started_at_utc": result.get("started_at_utc"),
            "completed_at_utc": result.get("completed_at_utc"),
            "records_completed": 24,
            "record_ledger_sha256": source_validation["completed_record_ledger_sha256"],
            "runtime_environment": dict(result["runtime_environment"]),
            "runner_sha256": OA1_PRE_AMENDMENT_RUNNER_SHA256,
        },
        {
            "segment_id": 2,
            "status": "IN_PROGRESS",
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "records_completed": 0,
            "runtime_environment": dict(runtime),
            "runner_sha256": actual_runner_sha,
        },
    ]
    result["runner_sha256"] = actual_runner_sha
    result["status"] = "SCOUT_IN_PROGRESS_OA1_RESUME"
    result["stop_reason"] = None
    result.pop("completed_at_utc", None)
    result["models"]["Qwen/Qwen3-4B"]["status"] = "RESUME_PENDING"
    _atomic_json(result_path, result)
    _render_figure(result, figure_path)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        for model_id, revision in MODEL_ORDER:
            model_entry = result["models"][model_id]
            if model_entry["status"] == "COMPLETE":
                continue
            model_entry["status"] = "ACQUIRING"
            _atomic_json(result_path, result)
            print(f"R1.2 OA-1 opening {model_id}: acquiring exact revision", flush=True)
            snapshot, inventory = r11._acquire_model(model_id, revision, cache_root)
            existing_inventory = model_entry.get("snapshot_inventory")
            if existing_inventory is not None and existing_inventory != inventory:
                raise ScoutError("OA-1 model snapshot inventory differs from segment 1")
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot, use_fast=True, local_files_only=True, trust_remote_code=False
            )
            _verify_token_binding(model_id, tokenizer, cases)
            model_entry["snapshot_inventory"] = inventory
            model_entry["status"] = "LOADING"
            _atomic_json(result_path, result)
            print(f"R1.2 OA-1 opening {model_id}: loading BF16 SDPA on cuda:0", flush=True)
            load_started = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                snapshot,
                dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=False,
                attn_implementation="sdpa",
            )
            model.eval()
            torch.cuda.synchronize()
            model_entry.setdefault("model_load_seconds_by_segment", {})["2"] = (
                time.perf_counter() - load_started
            )
            model_entry["status"] = "RUNNING"
            passed_b = False
            for surface in SURFACES:
                surface_entry = model_entry["surfaces"].get(surface)
                if surface_entry is None:
                    surface_entry = {
                        "status": "OPENED_IN_PROGRESS", "cells": {}, "assessment": None,
                    }
                    model_entry["surfaces"][surface] = surface_entry
                for shot_count in SHOT_COUNTS:
                    shot_entry = surface_entry["cells"].setdefault(str(shot_count), {})
                    for condition in CONDITIONS:
                        if condition in shot_entry:
                            if not (
                                model_id == "Qwen/Qwen3-4B"
                                and surface == "ICL-P2"
                                and shot_count == 4
                                and condition == "correct"
                            ):
                                raise ScoutError("OA-1 encountered an unexpected precompleted cell")
                            print("R1.2 OA-1 preserving immutable segment-1 cell", flush=True)
                            continue
                        cell_cases = [
                            case for case in cases[surface]
                            if case.shot_count == shot_count and case.condition == condition
                        ]
                        records = [
                            _score_case(model, tokenizer, case, "cuda:0")
                            for case in cell_cases
                        ]
                        completed_ids = {
                            record["case_id"]
                            for old_model in result["models"].values()
                            for old_surface in old_model.get("surfaces", {}).values()
                            for old_shot in old_surface.get("cells", {}).values()
                            for old_cell in old_shot.values()
                            for record in old_cell.get("records", [])
                        }
                        new_ids = {record["case_id"] for record in records}
                        if completed_ids & new_ids:
                            raise ScoutError("OA-1 attempted to reexecute a completed case")
                        result["forwards_completed"] += len(records)
                        result["execution_segments"][1]["records_completed"] += len(records)
                        if result["forwards_completed"] > FORWARD_CEILING:
                            raise ScoutError("forward ceiling exceeded")
                        metrics = _cell_metrics(records)
                        shot_entry[condition] = {
                            "status": "OPENED_COMPLETE",
                            "cases": len(records),
                            "records": records,
                            "metrics": metrics,
                            "execution_segment": 2,
                        }
                        print(
                            f"R1.2 OA-1 {model_id} {surface} {shot_count} {condition}: "
                            f"{metrics['classification']}",
                            flush=True,
                        )
                        _atomic_json(result_path, result)
                        _render_figure(result, figure_path)
                surface_entry["assessment"] = _formation_assessment(surface_entry["cells"])
                surface_entry["status"] = "OPENED_COMPLETE"
                outcome = surface_entry["assessment"]["outcome"]
                print(f"R1.2 OA-1 {model_id} {surface}: {outcome}", flush=True)
                _atomic_json(result_path, result)
                _render_figure(result, figure_path)
                if outcome != "FORMATION_PASS":
                    model_entry["failure_surface"] = surface
                    break
                if surface == "ICL-B":
                    passed_b = True
            model_entry["status"] = "COMPLETE"
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            _atomic_json(result_path, result)
            _render_figure(result, figure_path)
            if passed_b:
                result["status"] = "SCOUT_PASS_ICL_B_FORMATION"
                result["stop_reason"] = (
                    f"first ICL-B FORMATION_PASS at {model_id}; larger models remain unopened"
                )
                break
        else:
            result["status"] = "SCOUT_CLOSE_NO_ICL_XOR_FORMATION_THROUGH_8B"
            result["stop_reason"] = (
                "Qwen3-4B/8B bounded input-output-only ICL XOR-formation path closed"
            )
        result["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        result["execution_segments"][1]["status"] = "COMPLETE"
        result["execution_segments"][1]["completed_at_utc"] = result["completed_at_utc"]
        result["unopened_models"] = [
            model_id for model_id, _ in MODEL_ORDER
            if result["models"][model_id]["status"] == "UNOPENED"
        ]
        _atomic_json(result_path, result)
        _render_figure(result, figure_path)
        return result
    except Exception as error:
        result["status"] = "OPERATIONAL_ABORT_OA1_FINAL"
        result["stop_reason"] = f"{type(error).__name__}: {error}"
        result["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        result["execution_segments"][1]["status"] = "OPERATIONAL_ABORT_FINAL"
        result["execution_segments"][1]["completed_at_utc"] = result["completed_at_utc"]
        _atomic_json(result_path, result)
        try:
            _render_figure(result, figure_path)
        except Exception:
            pass
        raise


def _modal_image(
    modal: Any, *, abort_result_path: Path, pre_amendment_runner: Path,
) -> Any:
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
        .add_local_dir(
            HERE, "/opt/graph_xor_r1_2", copy=True,
            ignore=["r1_2_results.json", "r1_2_formation_matrix.png", "**/__pycache__/**"],
        )
        .add_local_dir(
            R11_ROOT, "/opt/graph_xor_r1_1", copy=True,
            ignore=["**/__pycache__/**"],
        )
        .add_local_dir(
            V1_ROOT, "/opt/graph_xor_r1_v1", copy=True,
            ignore=["b0/rust/target/**", "**/__pycache__/**"],
        )
        .add_local_file(
            abort_result_path, "/opt/graph_xor_r1_2/oa1_abort_result.json", copy=True
        )
        .add_local_file(
            pre_amendment_runner,
            "/opt/graph_xor_r1_2/oa1_pre_amendment_runner.py",
            copy=True,
        )
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


def run_modal_campaign(
    plan_path: Path, result_path: Path, figure_path: Path,
    *, abort_result_path: Path, pre_amendment_runner: Path,
) -> dict[str, Any]:
    installed = importlib.metadata.version("modal")
    if installed != MODAL_SDK_VERSION:
        raise ScoutError(f"Modal SDK drift: {installed} != {MODAL_SDK_VERSION}")
    import modal

    plan_sha = _sha256_file(plan_path)
    runner_sha = _sha256_file(Path(__file__))
    plan_text = plan_path.read_text(encoding="utf-8")
    if plan_sha != OA1_PRE_AMENDMENT_PLAN_SHA256:
        raise ScoutError("OA-1 scientific plan hash changed")
    if f"RUNNER_SHA256: {OA1_PRE_AMENDMENT_RUNNER_SHA256}" not in plan_text:
        raise ScoutError("OA-1 plan lost its pre-amendment runner binding")
    if _sha256_file(abort_result_path) != OA1_ABORT_RESULT_SHA256:
        raise ScoutError("OA-1 local abort result binding changed")
    if _sha256_file(pre_amendment_runner) != OA1_PRE_AMENDMENT_RUNNER_SHA256:
        raise ScoutError("OA-1 local pre-amendment runner binding changed")
    if "TO_BE_COMPILED" in plan_text or TOKEN_BINDING.get("status") != "TOKENIZER_ONLY_COMPILE_PASS":
        raise ScoutError("one-page plan or runner lacks final exact binding")
    app = modal.App.lookup(MODAL_APP, create_if_missing=True)
    volume = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True, version=2)
    volume.hydrate()
    image = _modal_image(
        modal,
        abort_result_path=abort_result_path,
        pre_amendment_runner=pre_amendment_runner,
    )
    with modal.enable_output():
        image = image.build(app)
    sandbox = modal.Sandbox.create(
        app=app, image=image, gpu=GPU_REQUEST, cpu=4.0, memory=32768,
        timeout=CAMPAIGN_SECONDS_CEILING, workdir="/opt/graph_xor_r1_2",
        volumes={"/vol": volume},
        env={
            "GRAPH_XOR_R11_ROOT": "/opt/graph_xor_r1_1",
            "GRAPH_XOR_R1_B0_PATH": "/opt/graph_xor_r1_v1/b0",
            "GRAPH_XOR_R1_V1_ROOT": "/opt/graph_xor_r1_v1",
            "HF_HOME": "/vol/hf-home",
            "PYTHONUTF8": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        tags={"program": "graph_xor-r1-2", "type": "icl-formation", "campaign": "one-shot"},
    )
    run_id = "r1-2-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_root = f"/vol/r1_2/{run_id}"
    remote_result = f"{remote_root}/r1_2_results.json"
    remote_figure = f"{remote_root}/r1_2_formation_matrix.png"
    try:
        process = sandbox.exec(
            "python", "/opt/graph_xor_r1_2/run_r1_2.py", "resume-oa1",
            "--plan", "/opt/graph_xor_r1_2/R1_2_ICL_FORMATION.md",
            "--abort-result", "/opt/graph_xor_r1_2/oa1_abort_result.json",
            "--pre-amendment-runner", "/opt/graph_xor_r1_2/oa1_pre_amendment_runner.py",
            "--result", remote_result,
            "--figure", remote_figure,
            "--expected-plan-sha", plan_sha,
            "--expected-runner-sha", runner_sha,
            "--cache-root", "/vol/hf-home",
            timeout=CAMPAIGN_SECONDS_CEILING - 180,
            workdir="/opt/graph_xor_r1_2",
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
            raise ScoutError("downloaded result lost plan/runner binding")
        return result
    finally:
        try:
            sandbox.terminate(wait=False)
        finally:
            try:
                sandbox.detach()
            except Exception:
                pass


def _renderer_oa1_test() -> dict[str, Any]:
    high_metrics = {
        "classification": "BEHAVIOR_PASS", "semantic_accuracy": 1.0,
    }
    low_metrics = {
        "classification": "NO_DETECTED_SIGNAL", "semantic_accuracy": 0.0,
    }
    result: dict[str, Any] = {
        "status": "SYNTHETIC_RENDER_STATE",
        "forwards_completed": 0,
        "historical_zero_shot_reference": {},
        "models": {
            model_id: {"surfaces": {}} for model_id, _ in MODEL_ORDER
        },
    }
    four_b = result["models"]["Qwen/Qwen3-4B"]["surfaces"]
    four_b["ICL-P2"] = {"cells": {"4": {"correct": {"metrics": high_metrics}}}}
    four_b["ICL-P3"] = {
        "cells": {"4": {"correct": {"metrics": low_metrics}}},
        "assessment": None,
    }
    four_b["ICL-P5"] = {
        "cells": {"4": {"correct": {"metrics": high_metrics}}},
        "assessment": {"outcome": "FORMATION_PASS"},
    }
    four_b["ICL-B"] = {
        "cells": {"4": {"correct": {"metrics": low_metrics}}},
        "assessment": {"outcome": "NO_DETECTED_FORMATION"},
    }
    legacy_failures = []
    for value in (0.0, 1.0):
        state = {
            "assessment": None,
            "cells": {"4": {"correct": {"metrics": {"semantic_accuracy": value}}}},
        }
        try:
            state.get("assessment", {}).get("outcome", "UNOPENED")
        except AttributeError:
            legacy_failures.append(value)
    if legacy_failures != [0.0, 1.0]:
        raise ScoutError("legacy renderer failure was not value-independent")
    with tempfile.TemporaryDirectory(prefix="graph_xor-r12-render-") as directory:
        figure = Path(directory) / "fixture.png"
        _render_figure(result, figure)
        if not figure.is_file() or figure.stat().st_size == 0:
            raise ScoutError("OA-1 renderer fixture did not produce an image")
    return {
        "status": "PASS",
        "assessment_key_absent": True,
        "assessment_null": True,
        "assessment_completed": True,
        "mixed_completed_pending": True,
        "legacy_failure_value_independent": True,
    }


def self_test() -> dict[str, Any]:
    cases, banks = compile_eval_cases()
    ancestors = _verify_immutable_ancestors()
    perfect_records: list[dict[str, Any]] = []
    for case in [
        row for row in cases["ICL-P2"]
        if row.shot_count == 4 and row.condition == "correct"
    ]:
        score = 1.0 if case.semantic_answer else -1.0
        perfect_records.append({
            "pair_id": case.pair_id,
            "semantic_answer": case.semantic_answer,
            "semantic_score": score,
            "semantic_prediction": case.semantic_answer,
            "candidate_pair_id": "direct-01",
            "physical_choice_literal": "1" if case.semantic_answer else "0",
            "entry_position": None,
            "strata": dict(case.strata),
        })
    perfect = _cell_metrics(perfect_records)
    if perfect["classification"] != "BEHAVIOR_PASS":
        raise ScoutError("perfect synthetic record failed behavioral pass")
    dummy_cells: dict[str, Any] = {}
    for shot in SHOT_COUNTS:
        dummy_cells[str(shot)] = {
            "correct": {"metrics": dict(perfect)},
            "label_shuffled": {"metrics": {
                **perfect,
                "classification": "NO_DETECTED_SIGNAL",
                "semantic_accuracy": 0.5,
                "semantic_score_auc": 0.5,
                "paired_directional_consistency": 0.5,
            }},
        }
    formation = _formation_assessment(dummy_cells)
    if formation["outcome"] != "FORMATION_PASS" or formation["selected_shot_count"] != 4:
        raise ScoutError("synthetic formation contrast failed")
    return {
        "status": "SELF_TEST_PASS",
        "case_generation_seed": CASE_SEED,
        "surface_case_counts": {surface: len(cases[surface]) for surface in SURFACES},
        "demo_counts": {surface: len(banks[surface]["demos"]) for surface in SURFACES},
        "target_counts": {surface: len(banks[surface]["targets"]) for surface in SURFACES},
        "case_ledger_sha256": _sha256_bytes(_canonical_bytes(_case_ledger(cases, banks))),
        "worst_case_scored_forwards": 2 * sum(len(rows) for rows in cases.values()),
        "synthetic_behavior": perfect,
        "synthetic_formation": formation,
        "oa1_renderer_regression": _renderer_oa1_test(),
        "immutable_ancestor_bindings": ancestors,
    }


def _configure_console_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def main() -> None:
    _configure_console_utf8()
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
    resume_parser = sub.add_parser("resume-oa1")
    resume_parser.add_argument("--plan", type=Path, required=True)
    resume_parser.add_argument("--abort-result", type=Path, required=True)
    resume_parser.add_argument("--pre-amendment-runner", type=Path, required=True)
    resume_parser.add_argument("--result", type=Path, required=True)
    resume_parser.add_argument("--figure", type=Path, required=True)
    resume_parser.add_argument("--expected-plan-sha", required=True)
    resume_parser.add_argument("--expected-runner-sha", required=True)
    resume_parser.add_argument("--cache-root", type=Path, required=True)
    modal_parser = sub.add_parser("modal-campaign")
    modal_parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    modal_parser.add_argument("--result", type=Path, default=RESULT_PATH)
    modal_parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    modal_parser.add_argument("--abort-result", type=Path, default=RESULT_PATH)
    modal_parser.add_argument("--pre-amendment-runner", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        report = self_test()
    elif args.command == "compile-tokenizers":
        report = tokenizer_only_compile(args.cache_root)
        _atomic_json(args.output, report)
    elif args.command == "execute":
        report = execute_scout(
            plan_path=args.plan,
            result_path=args.result,
            figure_path=args.figure,
            expected_plan_sha=args.expected_plan_sha,
            expected_runner_sha=args.expected_runner_sha,
            cache_root=args.cache_root,
        )
    elif args.command == "resume-oa1":
        report = resume_scout_oa1(
            plan_path=args.plan,
            abort_result_path=args.abort_result,
            pre_amendment_runner=args.pre_amendment_runner,
            result_path=args.result,
            figure_path=args.figure,
            expected_plan_sha=args.expected_plan_sha,
            expected_runner_sha=args.expected_runner_sha,
            cache_root=args.cache_root,
        )
    else:
        report = run_modal_campaign(
            args.plan, args.result, args.figure,
            abort_result_path=args.abort_result,
            pre_amendment_runner=args.pre_amendment_runner,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
