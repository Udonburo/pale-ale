"""Fail-closed Track A runner; no model is loaded before dual-lock validation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.gate13_causal_return.phase2_common import read_json, sha256_bytes, write_json
from tools.gate13_causal_return.validate_phase2_locks import validate_phase2_locks

from .compile_phase2_cases import compile_cases, validate_manifests
from .compile_register_cases import compile_ledger
from .evaluate_phase2 import evaluate_a0, evaluate_a1, evaluate_a2


class TrackARuntimeError(RuntimeError):
    """Fail-closed runtime or execution error."""


def model_load_authorized(
    validation: Mapping[str, Any], probe: Mapping[str, Any], *, probe_only: bool
) -> bool:
    """Return true only when both the dual authorization and exact runtime pass."""
    if probe_only:
        return False
    return bool(validation.get("execution_authorized")) and probe.get("status") == "PASS"


def runtime_probe(lock: Mapping[str, Any]) -> dict[str, Any]:
    required = lock["runtime_binding"]
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "tokenizers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "UNAVAILABLE"
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda_available else "NONE"
        memory = (
            int(torch.cuda.get_device_properties(0).total_memory) if cuda_available else 0
        )
        cuda_runtime = str(torch.version.cuda) if torch.version.cuda else "NONE"
    except Exception as exc:  # pragma: no cover - defensive runtime boundary
        cuda_available = False
        gpu = "NONE"
        memory = 0
        cuda_runtime = "NONE"
        versions["torch_probe_error"] = type(exc).__name__
    try:
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()[0]
    except Exception:
        driver = "UNAVAILABLE"

    observed = {
        "python": platform.python_version(),
        "torch": versions["torch"],
        "transformers": versions["transformers"],
        "tokenizers": versions["tokenizers"],
        "cuda_available": cuda_available,
        "cuda_runtime": cuda_runtime,
        "nvidia_driver": driver,
        "gpu": gpu,
        "gpu_memory_bytes": memory,
    }
    checks = {
        "python": observed["python"] == required["python"],
        "torch": observed["torch"] == required["pytorch"],
        "transformers": observed["transformers"] == required["transformers"],
        "tokenizers": observed["tokenizers"] == required["tokenizers"],
        "cuda_available": cuda_available,
        "cuda_runtime": observed["cuda_runtime"] == required["cuda"],
        "driver": observed["nvidia_driver"] == required["driver"],
        "gpu": required["gpu"] in observed["gpu"],
        "gpu_memory": memory >= int(required["minimum_gpu_memory_bytes"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "BLOCKED_EXACT_RUNTIME_UNAVAILABLE",
        "observed": observed,
        "checks": checks,
    }


def _load_completed(path: Path, allowed_ids: set[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TrackARuntimeError("execution state contains a non-object row")
            records.append(value)
    ids = [str(row.get("case_id") or "") for row in records]
    if any(not case_id or case_id not in allowed_ids for case_id in ids):
        raise TrackARuntimeError("execution state contains an empty or unknown case_id")
    if len(ids) != len(set(ids)):
        raise TrackARuntimeError("execution state contains a duplicate case_id")
    return records


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _load_attempts(path: Path, allowed_ids: set[str]) -> list[dict[str, Any]]:
    attempts = _load_completed(path, allowed_ids)
    ids = [str(row["case_id"]) for row in attempts]
    if len(ids) != len(set(ids)):
        raise TrackARuntimeError("a case has more than one scientific attempt")
    return attempts


def _load_exact_model(lock: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runtime = lock["runtime_binding"]
    tokenizer = AutoTokenizer.from_pretrained(
        runtime["model_repository"],
        revision=runtime["tokenizer_revision"],
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    template_hash = sha256_bytes(str(tokenizer.chat_template).encode("utf-8"))
    if template_hash != runtime["chat_template_sha256"]:
        raise TrackARuntimeError("chat template SHA-256 mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        runtime["model_repository"],
        revision=runtime["model_revision"],
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    return torch, tokenizer, model


def _generate_one(torch: Any, tokenizer: Any, model: Any, prompt: str, max_tokens: int) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to("cuda:0")
    with torch.inference_mode():
        output = model.generate(
            inputs,
            do_sample=False,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    continuation = output[0, inputs.shape[-1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True)


def _execute_stage(
    *,
    stage: str,
    cases: Sequence[Mapping[str, Any]],
    state_path: Path,
    torch: Any,
    tokenizer: Any,
    model: Any,
    forward_count: int,
    ceiling: int,
) -> tuple[list[dict[str, Any]], int]:
    allowed = {str(case["case_id"]) for case in cases}
    records = _load_completed(state_path, allowed)
    completed = {str(record["case_id"]) for record in records}
    attempt_path = state_path.with_name(state_path.stem + "_attempts.jsonl")
    attempts = _load_attempts(attempt_path, allowed)
    attempted = {str(record["case_id"]) for record in attempts}
    incomplete = attempted - completed
    if incomplete:
        raise TrackARuntimeError(
            "a prior case-level model call has no saved response; scientific retry is forbidden"
        )
    if completed - attempted:
        raise TrackARuntimeError("saved response lacks its immutable attempt journal entry")
    forward_count += len(attempts)
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in completed:
            continue
        if forward_count >= ceiling:
            raise TrackARuntimeError("forward ceiling reached before stage completion")
        forward_count += 1
        _append_record(
            attempt_path,
            {
                "case_id": case_id,
                "stage": stage,
                "case_level_model_forward": forward_count,
                "status": "STARTED_NO_RETRY",
            },
        )
        response = _generate_one(
            torch,
            tokenizer,
            model,
            str(case["prompt"]),
            max_tokens=max(32, len(str(case["expected_text"])) // 2 + 32),
        )
        record = {
            "case_id": case_id,
            "stage": stage,
            "response": response,
            "case_level_model_forward": forward_count,
        }
        _append_record(state_path, record)
        records.append(record)
    return records, forward_count


def run_track_a(
    *,
    phase2_dir: Path,
    output_dir: Path,
    probe_only: bool = False,
) -> dict[str, Any]:
    validation = validate_phase2_locks(phase2_dir=phase2_dir, require_clean=True)
    if validation["status"] != "PASS":
        raise TrackARuntimeError("dual-lock validation did not pass")
    lock = read_json(phase2_dir / "phase2_a_lock.json")
    a0_extension_manifest = read_json(
        phase2_dir / lock["case_manifests"]["A0_EXTENSION"]["path"]
    )
    a1_manifest = read_json(phase2_dir / lock["case_manifests"]["A1"]["path"])
    a2_manifest = read_json(phase2_dir / lock["case_manifests"]["A2"]["path"])
    validate_manifests(a1_manifest, a2_manifest, a0_extension_manifest)
    probe = runtime_probe(lock)
    if probe["status"] != "PASS":
        result = {
            "schema_version": "gate13_phase2_track_a_result_v1",
            "status": "READY_FOR_EXTERNAL_EXECUTION",
            "runtime_binding_status": "BLOCKED_EXACT_RUNTIME_UNAVAILABLE",
            "runtime_probe": probe,
            "TRACK_A_A0": "UNOPENED",
            "TRACK_A_A1": "UNOPENED",
            "TRACK_A_A2": "UNOPENED",
            "model_forward_count": 0,
            "activation_extraction_count": 0,
            "dual_execution_authorized": bool(validation["execution_authorized"]),
        }
        write_json(output_dir / "track_a_result.json", result)
        return result
    if probe_only:
        return {"status": "PASS_EXACT_RUNTIME_AVAILABLE", "runtime_probe": probe}
    if not model_load_authorized(validation, probe, probe_only=False):
        result = {
            "schema_version": "gate13_phase2_track_a_result_v1",
            "status": "NOT_EXECUTED_DUAL_AUTHORIZATION_BLOCKED",
            "runtime_binding_status": "PASS_EXACT_RUNTIME_AVAILABLE",
            "runtime_probe": probe,
            "TRACK_A_A0": "UNOPENED",
            "TRACK_A_A1": "UNOPENED",
            "TRACK_A_A2": "UNOPENED",
            "model_forward_count": 0,
            "activation_extraction_count": 0,
            "dual_execution_authorized": False,
        }
        write_json(output_dir / "track_a_result.json", result)
        return result

    torch, tokenizer, model = _load_exact_model(lock)
    compiled = compile_cases()
    a0_cases = list(compile_ledger()["cases"]) + list(compiled["A0_EXTENSION"])
    forward_count = 0
    a0_records, forward_count = _execute_stage(
        stage="A0",
        cases=a0_cases,
        state_path=output_dir / "a0_state.jsonl",
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        forward_count=forward_count,
        ceiling=int(lock["forward_ceiling"]),
    )
    a0 = evaluate_a0(a0_cases, a0_records)
    result: dict[str, Any] = {
        "schema_version": "gate13_phase2_track_a_result_v1",
        "status": "MANDATORY_STOP",
        "runtime_probe": probe,
        "TRACK_A_A0": a0["status"],
        "TRACK_A_A1": "UNOPENED",
        "TRACK_A_A2": "UNOPENED",
        "A0": a0,
        "model_forward_count": forward_count,
        "activation_extraction_count": 0,
    }
    if a0["status"] == "PASS":
        a1_records, forward_count = _execute_stage(
            stage="A1",
            cases=compiled["A1"],
            state_path=output_dir / "a1_state.jsonl",
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            forward_count=forward_count,
            ceiling=int(lock["forward_ceiling"]),
        )
        a1 = evaluate_a1(compiled["A1"], a1_records)
        result.update({"TRACK_A_A1": a1["status"], "A1": a1})
        if a1["status"] == "PASS":
            a2_records, forward_count = _execute_stage(
                stage="A2",
                cases=compiled["A2"],
                state_path=output_dir / "a2_state.jsonl",
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                forward_count=forward_count,
                ceiling=int(lock["forward_ceiling"]),
            )
            a2 = evaluate_a2(compiled["A2"], a2_records)
            result.update({"TRACK_A_A2": a2["status"], "A2": a2})
    result["model_forward_count"] = forward_count
    write_json(output_dir / "track_a_result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_track_a(
        phase2_dir=args.phase2_dir,
        output_dir=args.output_dir,
        probe_only=args.probe_only,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
