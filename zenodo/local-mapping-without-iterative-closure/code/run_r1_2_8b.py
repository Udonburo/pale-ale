"""Fresh Qwen3-8B-only ICL XOR/parity formation-boundary scout."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import gc
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
R12_ROOT = Path(os.environ.get("GRAPH_XOR_R12_ROOT", str(HERE.parent / "graph_xor_r1_2")))
R11_ROOT = Path(os.environ.get("GRAPH_XOR_R11_ROOT", str(HERE.parent / "graph_xor_r1_1")))
V1_ROOT = Path(os.environ.get("GRAPH_XOR_R1_V1_ROOT", str(HERE.parent / "graph_xor_r1")))
os.environ.setdefault("GRAPH_XOR_R11_ROOT", str(R11_ROOT))
os.environ.setdefault("GRAPH_XOR_R1_V1_ROOT", str(V1_ROOT))
os.environ.setdefault("GRAPH_XOR_R1_B0_PATH", str(V1_ROOT / "b0"))

_base_spec = importlib.util.spec_from_file_location("graph_xor_r1_2_frozen", R12_ROOT / "run_r1_2.py")
if _base_spec is None or _base_spec.loader is None:
    raise RuntimeError("could not load frozen R1.2 runner")
base = importlib.util.module_from_spec(_base_spec)
sys.modules[_base_spec.name] = base
_base_spec.loader.exec_module(base)


PLAN_PATH = HERE / "R1_2_8B_FORMATION_BOUNDARY.md"
RESULT_PATH = HERE / "r1_2_8b_results.json"
FIGURE_PATH = HERE / "r1_2_8b_formation_matrix.png"
SEED = 202_608_172_137
SURFACES = ("ICL-P2", "ICL-P3")
SHOT_COUNTS = (4, 16, 64)
CONDITIONS = ("correct", "label_shuffled")
TARGET_CASES = 24
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
FORWARD_CEILING = 300
MODEL_CONTEXT = 40_960
GPU_REQUEST = "L40S"
GPU_MEMORY_MIN_BYTES = 40 * 1024**3
CAMPAIGN_SECONDS_CEILING = 7_200
MODAL_APP = "graph_xor-r1-2-8b-formation-boundary"
REMOTE_VOLUME = base.MODAL_VOLUME
TOKENIZER_FILES = tuple(base.r11.TOKENIZER_FILES)
EXPECTED_R12 = {
    "plan_sha256": "d749b4b20a6d0d2978a82bef72e1905eecda1a316a5337f88ee4c0c87929b83b",
    "runner_sha256": "62c464dcfd44029467cf33db7207c1f223c19707cf633b5dd394eb40582e56f1",
    "result_sha256": "48979444b3a984de608a8c5c038932aa567390be7372b2ba1b8b8dc5ea66dc8d",
    "figure_sha256": "1c869f59034ff37e7d34baf4f0bdb361fe241c9276c30b8b076a8e0fceed9bc2",
}

# Frozen before model execution by the tokenizer-only compiler.
TOKEN_BINDING: dict[str, Any] = json.loads(r'''{"aggregate_sha256":"ef1c17c4dc82cbaa303b3963b759e8593cd1050a3126749467d99092bde2de19","case_ledger_sha256":"36c03f6b3ee289866cfbfc9524803541294cbf73b75494945ba7c6058606249d","cases_audited":288,"chat_template_sha256":"a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8","context_margin_positions":36164,"correct_shuffled_exact_token_count_and_multiset_match":true,"files":[{"name":"config.json","sha256":"f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30","size_bytes":728},{"name":"merges.txt","sha256":"8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5","size_bytes":1671853},{"name":"tokenizer.json","sha256":"aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4","size_bytes":11422654},{"name":"tokenizer_config.json","sha256":"d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101","size_bytes":9732},{"name":"vocab.json","sha256":"ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910","size_bytes":2776833}],"literal_token_ids":{"0":15,"1":16},"maximum_prefix_token_positions":4795,"model_id":"Qwen/Qwen3-8B","prefix_aggregate_sha256":"52e97c1ed5ee103cb45f68bb522e7ee1285f7e7e163b45f446d5fd288c20234e","revision":"b968826d9c46dd6066d109eabc6255188de91218","runtime":{"huggingface_hub":"1.27.0","jinja2":"3.1.6","tokenizers":"0.22.2","transformers":"5.15.0"},"status":"TOKENIZER_ONLY_COMPILE_PASS"}''')


class ScoutError(RuntimeError):
    """Fail-closed scout error."""


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_u64(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(_canonical_bytes(parts)).digest()[:8], "big")


def _rng(*parts: object) -> random.Random:
    return random.Random(_stable_u64("graph_xor-r1-2-8b", SEED, *parts))


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def _build_p2() -> tuple[list[Any], list[Any]]:
    demos: list[Any] = []
    combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for block in range(16):
        order = list(combinations)
        _rng("p2-demo-order", block).shuffle(order)
        for slot, (first, second) in enumerate(order):
            index = block * 4 + slot
            answer = first ^ second
            demos.append(base._item(
                surface="ICL-P2", item_id=f"F8-P2-demo-{index:03d}",
                pair_id=f"F8-P2-demo-{index:03d}", semantic_answer=answer,
                body=base._p2_body(f"f8d{index:03d}", first, second),
                strata={"semantic_answer": answer},
                metadata={"edge_bits": [first, second], "role": "demo", "branch": "fresh-8b"},
            ))
    targets: list[Any] = []
    for pair_index in range(12):
        first = pair_index % 2
        for answer in (0, 1):
            second = first ^ answer
            targets.append(base._item(
                surface="ICL-P2", item_id=f"F8-P2-target-{pair_index:02d}-{answer}",
                pair_id=f"F8-P2-target-{pair_index:02d}", semantic_answer=answer,
                body=base._p2_body(f"f8t{pair_index:02d}{answer}", first, second),
                strata={"semantic_answer": answer, "first_edge": first},
                metadata={"edge_bits": [first, second], "role": "target", "branch": "fresh-8b"},
            ))
    return demos, targets


def _fresh_p3_prefixes() -> list[tuple[int, ...]]:
    historical = {
        tuple(case.metadata["bits"][:-1])
        for case in base.r11.compile_cases()["P3"]
    }
    old = base._build_p3_p5()["ICL-P3"]
    historical.update(tuple(item.metadata["bits"][:-1]) for role in old for item in role)
    candidates = [tuple(map(int, f"{value:07b}")) for value in range(128)]
    candidates = [candidate for candidate in candidates if candidate not in historical]
    _rng("fresh-p3-prefixes").shuffle(candidates)
    if len(candidates) < 44:
        raise ScoutError(f"only {len(candidates)} unused length-7 prefixes remain")
    return candidates[:44]


def _build_p3() -> tuple[list[Any], list[Any]]:
    prefixes = _fresh_p3_prefixes()
    demos: list[Any] = []
    targets: list[Any] = []
    for pair_index, prefix in enumerate(prefixes[:32]):
        for answer in (0, 1):
            bits = base._bits_for_prefix(prefix, answer)
            demos.append(base._item(
                surface="ICL-P3", item_id=f"F8-P3-demo-{pair_index:02d}-{answer}",
                pair_id=f"F8-P3-demo-{pair_index:02d}", semantic_answer=answer,
                body=base._p3_body(bits),
                strata={"semantic_answer": answer, "parity_length": 8},
                metadata={"bits": list(bits), "role": "demo", "branch": "fresh-8b"},
            ))
    for pair_index, prefix in enumerate(prefixes[32:44]):
        for answer in (0, 1):
            bits = base._bits_for_prefix(prefix, answer)
            targets.append(base._item(
                surface="ICL-P3", item_id=f"F8-P3-target-{pair_index:02d}-{answer}",
                pair_id=f"F8-P3-target-{pair_index:02d}", semantic_answer=answer,
                body=base._p3_body(bits),
                strata={"semantic_answer": answer, "parity_length": 8},
                metadata={"bits": list(bits), "role": "target", "branch": "fresh-8b"},
            ))
    return demos, targets


def _shuffled_labels(demos: Sequence[Any]) -> list[int]:
    labels = [int(item.semantic_answer) for item in demos]
    shuffled: list[int] = []
    for start in range(0, len(labels), 4):
        block = labels[start:start + 4]
        if Counter(block) != {0: 2, 1: 2}:
            raise ScoutError("demo block is not balanced")
        options = []
        for permutation in itertools.permutations(range(4)):
            candidate = [block[index] for index in permutation]
            if sum(a != b for a, b in zip(block, candidate)) == 2 and candidate != block and candidate != [1 - value for value in block]:
                options.append(candidate)
        shuffled.extend(options[_stable_u64("fresh-8b-shuffle", SEED, start, block) % len(options)])
    return shuffled


def compile_cases() -> tuple[dict[str, list[Any]], dict[str, dict[str, list[Any]]]]:
    p2 = _build_p2()
    p3 = _build_p3()
    banks = {
        "ICL-P2": {"demos": p2[0], "targets": p2[1]},
        "ICL-P3": {"demos": p3[0], "targets": p3[1]},
    }
    result: dict[str, list[Any]] = {surface: [] for surface in SURFACES}
    for surface in SURFACES:
        demos = banks[surface]["demos"]
        correct = [int(item.semantic_answer) for item in demos]
        shuffled = _shuffled_labels(demos)
        for shot_count in SHOT_COUNTS:
            for condition, labels in (("correct", correct), ("label_shuffled", shuffled)):
                for target in banks[surface]["targets"]:
                    prompt = base._icl_prompt(demos[:shot_count], labels[:shot_count], target)
                    identity = {
                        "surface": surface, "shot_count": shot_count,
                        "condition": condition, "target_id": target.item_id,
                        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                    }
                    result[surface].append(base.EvalCase(
                        case_id=_sha256_bytes(_canonical_bytes(("graph_xor-r1-2-8b-case", identity))),
                        surface=surface, shot_count=shot_count, condition=condition,
                        pair_id=target.pair_id, semantic_answer=target.semantic_answer,
                        prompt=prompt, strata=dict(target.strata),
                        metadata={
                            "target_item_id": target.item_id,
                            "demo_item_ids": [item.item_id for item in demos[:shot_count]],
                            "demo_labels_sha256": _sha256_bytes(_canonical_bytes(labels[:shot_count])),
                            "target_metadata": dict(target.metadata),
                        },
                    ))
    _validate_cases(result, banks)
    return result, banks


def _validate_cases(cases: Mapping[str, Sequence[Any]], banks: Mapping[str, Mapping[str, Sequence[Any]]]) -> None:
    all_ids: set[str] = set()
    for surface in SURFACES:
        demos = list(banks[surface]["demos"])
        targets = list(banks[surface]["targets"])
        if len(demos) != 64 or len(targets) != 24:
            raise ScoutError(f"{surface} bank size drift")
        if set(item.body for item in demos) & set(item.body for item in targets):
            raise ScoutError(f"{surface} demo/target overlap")
        for shot in SHOT_COUNTS:
            if Counter(item.semantic_answer for item in demos[:shot]) != {0: shot // 2, 1: shot // 2}:
                raise ScoutError(f"{surface} {shot}-shot prefix imbalance")
            for condition in CONDITIONS:
                cell = [case for case in cases[surface] if case.shot_count == shot and case.condition == condition]
                if len(cell) != 24 or Counter(case.semantic_answer for case in cell) != {0: 12, 1: 12}:
                    raise ScoutError(f"{surface} {shot} {condition} target imbalance")
        ids = [case.case_id for case in cases[surface]]
        if len(ids) != 144 or len(ids) != len(set(ids)) or set(ids) & all_ids:
            raise ScoutError("fresh case ID collision")
        all_ids.update(ids)
        pairs = defaultdict(list)
        for target in targets:
            pairs[target.pair_id].append(target.semantic_answer)
        if len(pairs) != 12 or any(sorted(values) != [0, 1] for values in pairs.values()):
            raise ScoutError(f"{surface} lost exact matched pairs")
        shuffled = _shuffled_labels(demos)
        correct = [item.semantic_answer for item in demos]
        for shot in SHOT_COUNTS:
            if Counter(shuffled[:shot]) != Counter(correct[:shot]) or sum(a != b for a, b in zip(shuffled[:shot], correct[:shot])) != shot // 2:
                raise ScoutError(f"{surface} {shot}-shot shuffled control drift")
    old_cases, _ = base.compile_eval_cases()
    old_ids = {case.case_id for surface in SURFACES for case in old_cases[surface]}
    if all_ids & old_ids:
        raise ScoutError("fresh 8B ledger overlaps frozen R1.2 case IDs")
    if sum(len(rows) for rows in cases.values()) != 288:
        raise ScoutError("forward ledger is not exactly 288")


def _public_item(item: Any) -> dict[str, Any]:
    return base._public_item(item)


def _public_case(case: Any) -> dict[str, Any]:
    return base._public_case(case)


def _case_ledger(cases: Mapping[str, Sequence[Any]], banks: Mapping[str, Mapping[str, Sequence[Any]]]) -> dict[str, Any]:
    return {
        "banks": {surface: {role: [_public_item(item) for item in banks[surface][role]] for role in ("demos", "targets")} for surface in SURFACES},
        "eval_cases": {surface: [_public_case(case) for case in cases[surface]] for surface in SURFACES},
    }


def tokenizer_only_compile(cache_root: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    cases, banks = compile_cases()
    snapshot = Path(snapshot_download(
        repo_id=MODEL_ID, revision=MODEL_REVISION,
        allow_patterns=list(TOKENIZER_FILES), cache_dir=str(cache_root),
    ))
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    names = {path.relative_to(snapshot).as_posix() for path in files}
    if names != set(TOKENIZER_FILES):
        raise ScoutError(f"tokenizer inventory mismatch: {sorted(names)}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=True, trust_remote_code=False)
    literal_ids: dict[str, set[int]] = defaultdict(set)
    prefix_records: list[dict[str, Any]] = []
    tokens: dict[tuple[str, int, str, str], list[int]] = {}
    maximum = 0
    for surface in SURFACES:
        for case in cases[surface]:
            text, ids = base.r11._chat_prefix(tokenizer, case.prompt)
            maximum = max(maximum, len(ids))
            if len(ids) + 1 > MODEL_CONTEXT:
                raise ScoutError("prompt exceeds context window")
            for literal in ("0", "1"):
                literal_ids[literal].add(base.r11._appended_token_id(tokenizer, text, ids, literal))
            prefix_records.append({"case_id": case.case_id, "token_ids_sha256": _sha256_bytes(_canonical_bytes(ids)), "token_positions": len(ids)})
            tokens[(surface, case.shot_count, case.metadata["target_item_id"], case.condition)] = ids
    for surface in SURFACES:
        for shot in SHOT_COUNTS:
            for target in banks[surface]["targets"]:
                correct = tokens[(surface, shot, target.item_id, "correct")]
                shuffled = tokens[(surface, shot, target.item_id, "label_shuffled")]
                if len(correct) != len(shuffled) or Counter(correct) != Counter(shuffled):
                    raise ScoutError("correct/shuffled token matching failed")
    if any(len(values) != 1 for values in literal_ids.values()):
        raise ScoutError("context-dependent direct token IDs")
    binding = {
        "status": "TOKENIZER_ONLY_COMPILE_PASS",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [{"name": path.relative_to(snapshot).as_posix(), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size} for path in files],
        "chat_template_sha256": _sha256_bytes(str(tokenizer.chat_template).encode("utf-8")),
        "literal_token_ids": {literal: next(iter(values)) for literal, values in sorted(literal_ids.items())},
        "prefix_aggregate_sha256": _sha256_bytes(_canonical_bytes(prefix_records)),
        "case_ledger_sha256": _sha256_bytes(_canonical_bytes(_case_ledger(cases, banks))),
        "maximum_prefix_token_positions": maximum,
        "context_margin_positions": MODEL_CONTEXT - maximum - 1,
        "cases_audited": len(prefix_records),
        "correct_shuffled_exact_token_count_and_multiset_match": True,
        "runtime": {key: importlib.metadata.version(package) for key, package in (("transformers", "transformers"), ("tokenizers", "tokenizers"), ("huggingface_hub", "huggingface-hub"), ("jinja2", "jinja2"))},
    }
    binding["aggregate_sha256"] = _sha256_bytes(_canonical_bytes(binding))
    return binding


def _verify_token_binding(tokenizer: Any, cases: Mapping[str, Sequence[Any]]) -> None:
    if TOKEN_BINDING.get("status") != "TOKENIZER_ONLY_COMPILE_PASS":
        raise ScoutError("token binding is not frozen")
    if _sha256_bytes(str(tokenizer.chat_template).encode("utf-8")) != TOKEN_BINDING["chat_template_sha256"]:
        raise ScoutError("chat template drift")
    literal_ids: dict[str, set[int]] = defaultdict(set)
    prefix_records: list[dict[str, Any]] = []
    for surface in SURFACES:
        for case in cases[surface]:
            text, ids = base.r11._chat_prefix(tokenizer, case.prompt)
            for literal in ("0", "1"):
                literal_ids[literal].add(base.r11._appended_token_id(tokenizer, text, ids, literal))
            prefix_records.append({"case_id": case.case_id, "token_ids_sha256": _sha256_bytes(_canonical_bytes(ids)), "token_positions": len(ids)})
    actual = {literal: next(iter(values)) for literal, values in literal_ids.items() if len(values) == 1}
    if actual != TOKEN_BINDING["literal_token_ids"]:
        raise ScoutError("direct token binding drift")
    if _sha256_bytes(_canonical_bytes(prefix_records)) != TOKEN_BINDING["prefix_aggregate_sha256"]:
        raise ScoutError("prompt-prefix ledger drift")


def _verify_ancestors() -> dict[str, Any]:
    ancestors = base._verify_immutable_ancestors()
    actual = {
        "plan_sha256": _sha256_file(R12_ROOT / "R1_2_ICL_FORMATION.md"),
        "runner_sha256": _sha256_file(R12_ROOT / "run_r1_2.py"),
        "result_sha256": _sha256_file(R12_ROOT / "r1_2_results.json"),
        "figure_sha256": _sha256_file(R12_ROOT / "r1_2_formation_matrix.png"),
    }
    if actual != EXPECTED_R12:
        raise ScoutError(f"frozen R1.2 binding drift: {actual}")
    result = json.loads((R12_ROOT / "r1_2_results.json").read_text(encoding="utf-8"))
    if result.get("status") != "OPERATIONAL_ABORT_OA1_FINAL" or result.get("forwards_completed") != 288:
        raise ScoutError("R1.2 is not in its frozen partial-close state")
    return {**ancestors, "r1_2": actual}


def _verify_runtime() -> dict[str, Any]:
    import torch
    runtime = base._runtime_environment()
    for key, expected in base.RUNTIME_LOCK.items():
        if runtime.get(key) != expected:
            raise ScoutError(f"runtime drift for {key}: {runtime.get(key)!r} != {expected!r}")
    if not runtime["cuda_available"] or "L40S" not in str(runtime["cuda_device_name"]).upper():
        raise ScoutError(f"did not receive frozen L40S: {runtime['cuda_device_name']}")
    if int(runtime["cuda_device_total_memory_bytes"]) < GPU_MEMORY_MIN_BYTES:
        raise ScoutError("GPU memory below 40 GiB")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return runtime


def _render_figure(result: Mapping[str, Any], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    width, height = 920, 300
    image = Image.new("RGB", (width, height), "#101318")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((18, 14), "GRAPH_XOR R1.2-8B Fresh Formation Boundary", fill="white", font=font)
    draw.text((18, 34), f"status: {result.get('status')} | forwards: {result.get('forwards_completed', 0)}", fill="#d2d8e2", font=font)
    columns = [("4C", 4, "correct"), ("4S", 4, "label_shuffled"), ("16C", 16, "correct"), ("16S", 16, "label_shuffled"), ("64C", 64, "correct"), ("64S", 64, "label_shuffled"), ("OUT", None, None)]
    colors = {"BEHAVIOR_PASS": "#1ea95a", "SCORE_SIGNAL_ONLY": "#e5a93d", "NO_DETECTED_SIGNAL": "#d95055", "FORMATION_PASS": "#1ea95a", "FORMATION_SIGNAL_ONLY": "#e5a93d", "NO_DETECTED_FORMATION": "#c43f50", "UNOPENED": "#434a55"}
    for index, (label, _, _) in enumerate(columns):
        draw.text((230 + index * 95, 68), label, fill="#dce2eb", font=font)
    for row_index, surface in enumerate(SURFACES):
        y = 90 + row_index * 78
        draw.text((18, y + 23), surface, fill="#e8edf5", font=font)
        entry = result.get("surfaces", {}).get(surface, {})
        for col_index, (_, shot, condition) in enumerate(columns):
            if shot is None:
                assessment = entry.get("assessment") or {}
                classification = assessment.get("outcome", "UNOPENED")
                label = {"FORMATION_PASS": "FORM", "FORMATION_SIGNAL_ONLY": "SIGNAL", "NO_DETECTED_FORMATION": "NONE"}.get(classification, "-")
            else:
                cell = entry.get("cells", {}).get(str(shot), {}).get(condition, {})
                metrics = cell.get("metrics", {})
                classification = metrics.get("classification", "UNOPENED")
                value = metrics.get("semantic_accuracy")
                label = "-" if value is None else f"{classification.split('_')[0]} {value:.0%}"
            x = 220 + col_index * 95
            draw.rectangle((x, y, x + 88, y + 58), fill=colors.get(classification, "#434a55"), outline="#a8b0bd")
            draw.text((x + 5, y + 22), label, fill="white", font=font)
    draw.text((18, 260), "green pass | amber score signal | red no detected signal | gray unopened", fill="#b9c0cc", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def execute(plan_path: Path, result_path: Path, figure_path: Path, expected_plan_sha: str, expected_runner_sha: str, cache_root: Path) -> dict[str, Any]:
    actual_plan = _sha256_file(plan_path)
    actual_runner = _sha256_file(Path(__file__))
    if actual_plan != expected_plan_sha or actual_runner != expected_runner_sha:
        raise ScoutError("plan/runner authorization hash mismatch")
    plan_text = plan_path.read_text(encoding="utf-8")
    if f"RUNNER_SHA256: {actual_runner}" not in plan_text or "TO_BE_COMPILED" in plan_text:
        raise ScoutError("one-page plan is not exactly frozen")
    cases, banks = compile_cases()
    ledger_sha = _sha256_bytes(_canonical_bytes(_case_ledger(cases, banks)))
    if ledger_sha != TOKEN_BINDING.get("case_ledger_sha256"):
        raise ScoutError("case ledger differs from tokenizer binding")
    runtime = _verify_runtime()
    ancestors = _verify_ancestors()
    result: dict[str, Any] = {
        "schema_version": "1.0", "program_id": "GRAPH_XOR_R1_2_8B_FORMATION_BOUNDARY",
        "status": "SCOUT_IN_PROGRESS", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan_sha256": actual_plan, "runner_sha256": actual_runner,
        "case_generation_seed": SEED, "case_ledger_sha256": ledger_sha,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "runtime_environment": runtime, "immutable_ancestor_bindings": ancestors,
        "token_binding": TOKEN_BINDING, "forward_ceiling": FORWARD_CEILING,
        "forwards_completed": 0, "surfaces": {}, "stop_reason": None,
        "resume_authorized": False,
    }
    _atomic_json(result_path, result)
    _render_figure(result, figure_path)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        snapshot, inventory = base.r11._acquire_model(MODEL_ID, MODEL_REVISION, cache_root)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=True, trust_remote_code=False)
        _verify_token_binding(tokenizer, cases)
        result["snapshot_inventory"] = inventory
        result["status"] = "MODEL_LOADING"
        _atomic_json(result_path, result)
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, dtype=torch.bfloat16, device_map={"": "cuda:0"},
            low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=False,
            attn_implementation="sdpa",
        )
        model.eval(); torch.cuda.synchronize()
        result["model_load_seconds"] = time.perf_counter() - started
        result["status"] = "SCOUT_RUNNING"
        for surface in SURFACES:
            entry = {"status": "OPENED_IN_PROGRESS", "cells": {}, "assessment": None}
            result["surfaces"][surface] = entry
            for shot in SHOT_COUNTS:
                shot_entry: dict[str, Any] = {}
                entry["cells"][str(shot)] = shot_entry
                for condition in CONDITIONS:
                    cell_cases = [case for case in cases[surface] if case.shot_count == shot and case.condition == condition]
                    records = [base._score_case(model, tokenizer, case, "cuda:0") for case in cell_cases]
                    result["forwards_completed"] += len(records)
                    if result["forwards_completed"] > FORWARD_CEILING:
                        raise ScoutError("forward ceiling exceeded")
                    metrics = base._cell_metrics(records)
                    shot_entry[condition] = {"status": "OPENED_COMPLETE", "cases": len(records), "records": records, "metrics": metrics}
                    print(f"8B fresh {surface} {shot} {condition}: {metrics['classification']}", flush=True)
                    _atomic_json(result_path, result); _render_figure(result, figure_path)
            entry["assessment"] = base._formation_assessment(entry["cells"])
            entry["status"] = "OPENED_COMPLETE"
            print(f"8B fresh {surface}: {entry['assessment']['outcome']}", flush=True)
            _atomic_json(result_path, result); _render_figure(result, figure_path)
        p3_outcome = result["surfaces"]["ICL-P3"]["assessment"]["outcome"]
        if p3_outcome == "FORMATION_PASS":
            result["status"] = "SCOUT_PASS_8B_ICL_P3_FORMATION"
            result["stop_reason"] = "Qwen3-8B met the frozen length-8 parity formation criterion"
        else:
            result["status"] = "SCOUT_CLOSE_NO_8B_ICL_P3_FORMATION"
            result["stop_reason"] = "bounded Qwen3-8B input-output-only ICL parity line closed"
        result["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _atomic_json(result_path, result); _render_figure(result, figure_path)
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        return result
    except Exception as error:
        result["status"] = "OPERATIONAL_ABORT_FINAL_NO_RETRY"
        result["stop_reason"] = f"{type(error).__name__}: {error}"
        result["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _atomic_json(result_path, result)
        try: _render_figure(result, figure_path)
        except Exception: pass
        raise


def _modal_image(modal: Any) -> Any:
    packages = [
        "transformers==5.15.0", "tokenizers==0.22.2", "huggingface_hub==1.27.0",
        "jinja2==3.1.6", "accelerate==1.14.0", "safetensors==0.8.0",
        "pillow==11.3.0", "numpy==2.3.2",
    ]
    return (
        modal.Image.from_registry(base.IMAGE_BASE)
        .apt_install("libgomp1")
        .pip_install("torch==2.7.1", extra_index_url="https://download.pytorch.org/whl/cu126")
        .pip_install(*packages)
        .add_local_dir(HERE, "/opt/graph_xor_r1_2_8b", copy=True, ignore=["r1_2_8b_results.json", "r1_2_8b_formation_matrix.png", "**/__pycache__/**"])
        .add_local_dir(R12_ROOT, "/opt/graph_xor_r1_2", copy=True, ignore=["**/__pycache__/**"])
        .add_local_dir(R11_ROOT, "/opt/graph_xor_r1_1", copy=True, ignore=["**/__pycache__/**"])
        .add_local_dir(V1_ROOT, "/opt/graph_xor_r1_v1", copy=True, ignore=["b0/rust/target/**", "**/__pycache__/**"])
    )


def _stream_process(process: Any) -> tuple[int, str, str]:
    stdout_parts: list[str] = []; stderr_parts: list[str] = []
    def pump(stream: Any, destination: list[str], prefix: str) -> None:
        for line in stream:
            destination.append(line); print(f"[{prefix}] {line}", end="", flush=True)
    threads = [threading.Thread(target=pump, args=(process.stdout, stdout_parts, "remote"), daemon=True), threading.Thread(target=pump, args=(process.stderr, stderr_parts, "remote-error"), daemon=True)]
    for thread in threads: thread.start()
    code = process.wait()
    for thread in threads: thread.join()
    return code, "".join(stdout_parts), "".join(stderr_parts)


def run_modal_campaign(plan_path: Path, result_path: Path, figure_path: Path) -> dict[str, Any]:
    if importlib.metadata.version("modal") != base.MODAL_SDK_VERSION:
        raise ScoutError("Modal SDK version drift")
    import modal
    plan_sha = _sha256_file(plan_path); runner_sha = _sha256_file(Path(__file__))
    if f"RUNNER_SHA256: {runner_sha}" not in plan_path.read_text(encoding="utf-8") or "TO_BE_COMPILED" in plan_path.read_text(encoding="utf-8"):
        raise ScoutError("plan is not frozen to this runner")
    app = modal.App.lookup(MODAL_APP, create_if_missing=True)
    volume = modal.Volume.from_name(REMOTE_VOLUME, create_if_missing=True, version=2); volume.hydrate()
    image = _modal_image(modal)
    with modal.enable_output(): image = image.build(app)
    sandbox = modal.Sandbox.create(
        app=app, image=image, gpu=GPU_REQUEST, cpu=4.0, memory=32768,
        timeout=CAMPAIGN_SECONDS_CEILING, workdir="/opt/graph_xor_r1_2_8b",
        volumes={"/vol": volume},
        env={"GRAPH_XOR_R12_ROOT": "/opt/graph_xor_r1_2", "GRAPH_XOR_R11_ROOT": "/opt/graph_xor_r1_1", "GRAPH_XOR_R1_V1_ROOT": "/opt/graph_xor_r1_v1", "GRAPH_XOR_R1_B0_PATH": "/opt/graph_xor_r1_v1/b0", "HF_HOME": "/vol/hf-home", "PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false"},
        tags={"program": "graph_xor-r1-2-8b", "type": "fresh-formation-boundary", "campaign": "one-shot"},
    )
    run_id = "r1-2-8b-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_root = f"/vol/r1_2_8b/{run_id}"
    remote_result = f"{remote_root}/r1_2_8b_results.json"; remote_figure = f"{remote_root}/r1_2_8b_formation_matrix.png"
    try:
        process = sandbox.exec(
            "python", "/opt/graph_xor_r1_2_8b/run_r1_2_8b.py", "execute",
            "--plan", "/opt/graph_xor_r1_2_8b/R1_2_8B_FORMATION_BOUNDARY.md",
            "--result", remote_result, "--figure", remote_figure,
            "--expected-plan-sha", plan_sha, "--expected-runner-sha", runner_sha,
            "--cache-root", "/vol/hf-home", timeout=CAMPAIGN_SECONDS_CEILING - 180,
        )
        code, _, stderr = _stream_process(process)
        sync = sandbox.exec("sync", "/vol", timeout=120); sync.wait()
        if code != 0:
            try:
                sandbox.filesystem.copy_to_local(remote_result, str(result_path)); sandbox.filesystem.copy_to_local(remote_figure, str(figure_path))
            except Exception: pass
            raise ScoutError(f"remote campaign exited {code}: {stderr[-4000:]}")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        sandbox.filesystem.copy_to_local(remote_result, str(result_path)); sandbox.filesystem.copy_to_local(remote_figure, str(figure_path))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["plan_sha256"] != plan_sha or result["runner_sha256"] != runner_sha:
            raise ScoutError("downloaded result lost plan/runner binding")
        return result
    finally:
        try: sandbox.terminate(wait=False)
        finally:
            try: sandbox.detach()
            except Exception: pass


def self_test() -> dict[str, Any]:
    cases, banks = compile_cases()
    ledger_sha = _sha256_bytes(_canonical_bytes(_case_ledger(cases, banks)))
    ancestors = _verify_ancestors()
    return {
        "status": "SELF_TEST_PASS", "seed": SEED,
        "case_counts": {surface: len(cases[surface]) for surface in SURFACES},
        "scored_forwards": sum(len(rows) for rows in cases.values()),
        "case_ledger_sha256": ledger_sha,
        "fresh_from_r1_2": True, "immutable_ancestor_bindings": ancestors,
        "token_binding_status": TOKEN_BINDING.get("status"),
    }


def _configure_console_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure): reconfigure(encoding="utf-8", errors="backslashreplace")


def main() -> None:
    _configure_console_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    token = sub.add_parser("tokenizer-only-compile"); token.add_argument("--cache-root", type=Path, required=True); token.add_argument("--output", type=Path)
    execute_parser = sub.add_parser("execute")
    for name, kwargs in (("--plan", {"type": Path, "required": True}), ("--result", {"type": Path, "required": True}), ("--figure", {"type": Path, "required": True}), ("--expected-plan-sha", {"required": True}), ("--expected-runner-sha", {"required": True}), ("--cache-root", {"type": Path, "required": True})):
        execute_parser.add_argument(name, **kwargs)
    modal_parser = sub.add_parser("modal-campaign"); modal_parser.add_argument("--plan", type=Path, default=PLAN_PATH); modal_parser.add_argument("--result", type=Path, default=RESULT_PATH); modal_parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    args = parser.parse_args()
    if args.command == "self-test": report = self_test()
    elif args.command == "tokenizer-only-compile":
        report = tokenizer_only_compile(args.cache_root)
        if args.output: _atomic_json(args.output, report)
    elif args.command == "execute": report = execute(args.plan, args.result, args.figure, args.expected_plan_sha, args.expected_runner_sha, args.cache_root)
    else: report = run_modal_campaign(args.plan, args.result, args.figure)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
