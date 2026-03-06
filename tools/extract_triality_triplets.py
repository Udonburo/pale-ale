#!/usr/bin/env python3
"""Extract deterministic triality proxy triplets (V, Splus, Sminus) from a causal LM."""

import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
except ImportError as exc:  # pragma: no cover - user environment dependent
    raise RuntimeError(
        "This tool requires torch + transformers. Install with: "
        "python -m pip install torch transformers"
    ) from exc

PROJ_ID = "fwht_pad_pow2_take8_v1"
SPLUS_DEF_ID = "attn_lastlayer_weighted_hidden_v1"
SMINUS_DEF_ID_TEMPLATE = "lm_head_row_expectation_topk{topk}_v1"
NORM_EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract token-level triality proxy vectors (V, Splus, Sminus)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-file", help="UTF-8 prompt file path.")
    source.add_argument("--prompt", help="Prompt text.")

    target = parser.add_mutually_exclusive_group(required=False)
    target.add_argument("--target-answer", help="Teacher forcing target answer text.")
    target.add_argument("--target-answer-file", help="UTF-8 teacher forcing target answer file path.")

    parser.add_argument("--model-id", help="Optional explicit HF model id.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="runs/triality_smoke/triplets_out.ndjson")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic algorithms where possible.",
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0")
    if args.topk <= 0:
        parser.error("--topk must be > 0")
    return args


def read_prompt_text(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return str(args.prompt)
    path = Path(args.prompt_file)
    return path.read_text(encoding="utf-8")


def read_target_answer_text(args: argparse.Namespace) -> Optional[str]:
    if args.target_answer is not None:
        return str(args.target_answer)
    if args.target_answer_file is not None:
        return Path(args.target_answer_file).read_text(encoding="utf-8")
    return None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def discover_qwen35_candidates() -> List[str]:
    """Best-effort lookup of Qwen3.5 models in the 0.5B..1.5B range."""
    try:
        from huggingface_hub import HfApi  # type: ignore
    except Exception:
        return []
    try:
        api = HfApi()
        rows = api.list_models(search="Qwen3.5", limit=64)
    except Exception:
        return []

    out: List[str] = []
    pattern = re.compile(r"(0[._]?5|1[._]?5)b", re.IGNORECASE)
    for row in rows:
        model_id = getattr(row, "id", None)
        if not isinstance(model_id, str):
            continue
        lid = model_id.lower()
        if not lid.startswith("qwen/"):
            continue
        if "qwen3.5" not in lid and "qwen-3.5" not in lid and "qwen3_5" not in lid:
            continue
        if not pattern.search(lid):
            continue
        out.append(model_id)
    return sorted(set(out))


def build_model_candidates(model_id_override: Optional[str]) -> List[str]:
    if model_id_override:
        return [model_id_override]

    # Prefer requested order: Qwen2.5 1.5B -> Qwen2.5 0.5B -> any Qwen3.5 0.5B..1.5B
    ordered = [
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ]
    ordered.extend(discover_qwen35_candidates())

    seen = set()
    deduped: List[str] = []
    for item in ordered:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def resolve_model_revision(model_id: str, tokenizer: Any, model: Any) -> Optional[str]:
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if revision:
        return str(revision)
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if isinstance(init_kwargs, dict) and init_kwargs.get("_commit_hash"):
        return str(init_kwargs["_commit_hash"])

    # Best-effort fallback using hub metadata.
    try:
        from huggingface_hub import HfApi  # type: ignore

        info = HfApi().model_info(model_id)
        sha = getattr(info, "sha", None)
        if sha:
            return str(sha)
    except Exception:
        return None
    return None


def load_first_available_model(
    model_candidates: Sequence[str], device: torch.device
) -> Tuple[str, Any, Any, Optional[str]]:
    failures: List[str] = []
    for model_id in model_candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
            )
            if hasattr(model, "set_attn_implementation"):
                try:
                    model.set_attn_implementation("eager")
                except Exception:
                    pass
            model.eval()
            model.to(device)
            if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            revision = resolve_model_revision(model_id, tokenizer, model)
            return model_id, tokenizer, model, revision
        except Exception as exc:  # pragma: no cover - environment dependent
            failures.append(f"{model_id}: {exc}")
    joined = "\n".join(failures)
    raise RuntimeError(
        "Failed to load any candidate model. Tried:\n" + joined
    )


def ensure_finite_vector(values: Sequence[float], label: str) -> None:
    for idx, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label}[{idx}] is non-finite")


def fwht_inplace(values: List[float]) -> None:
    n = len(values)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("fwht_inplace requires power-of-two length")
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = values[j]
                y = values[j + h]
                values[j] = x + y
                values[j + h] = x - y
        h *= 2


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def normalize8(values: Sequence[float]) -> List[float]:
    if len(values) != 8:
        raise ValueError("normalize8 expects length 8")
    norm_sq = 0.0
    for value in values:
        fv = float(value)
        norm_sq += fv * fv
    norm = math.sqrt(norm_sq)
    if not math.isfinite(norm) or norm < NORM_EPS:
        # Deterministic fallback for near-zero vectors.
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = [float(v) / norm for v in values]
    ensure_finite_vector(out, "normalize8")
    return out


def project_fwht_to_8(values: Sequence[float]) -> List[float]:
    as_list = [float(v) for v in values]
    ensure_finite_vector(as_list, "project_input")
    length = len(as_list)
    if length <= 0:
        raise ValueError("cannot project empty vector")

    target_len = next_pow2(length)
    if target_len > length:
        as_list.extend([0.0] * (target_len - length))
    elif target_len < length:
        as_list = as_list[:target_len]

    fwht_inplace(as_list)
    out = as_list[:8]
    if len(out) < 8:
        out.extend([0.0] * (8 - len(out)))
    return normalize8(out)


def topk_probs_and_entropy(logits: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    vocab = int(logits.shape[0])
    k = min(max(1, topk), vocab)
    top_vals, top_idx = torch.topk(logits, k)
    top_probs = torch.softmax(top_vals, dim=0)
    entropy = float(
        (-(top_probs * torch.log(top_probs.clamp_min(1e-30))).sum()).item()
    )
    return top_idx, top_probs, entropy


def compute_splus_from_past(
    attn_to_past: torch.Tensor, hidden_past: torch.Tensor, hidden_dim: int
) -> torch.Tensor:
    # attn_to_past: [num_heads, t], hidden_past: [t, hidden]
    if int(hidden_past.shape[0]) <= 0:
        return torch.zeros(
            hidden_dim,
            device=hidden_past.device,
            dtype=hidden_past.dtype,
        )
    weights = attn_to_past.mean(dim=0)
    return torch.matmul(weights, hidden_past)


def select_target_token_indices_from_offsets(
    offsets: Sequence[Sequence[int]], answer_char_start: int
) -> List[int]:
    target_indices: List[int] = []
    for idx, pair in enumerate(offsets):
        if len(pair) < 2:
            continue
        start = int(pair[0])
        end = int(pair[1])
        # Exclude special tokens with empty offsets.
        if end <= start:
            continue
        # Token belongs to answer if it overlaps boundary or starts after it.
        if end > answer_char_start:
            target_indices.append(idx)
    return target_indices


def write_ndjson(path: Path, rows: Sequence[Dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            line = json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            payload = (line + "\n").encode("utf-8")
            hasher.update(payload)
            f.write(line + "\n")
    return hasher.hexdigest()


def write_meta_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")


def run_autoregressive_extraction(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int,
    topk: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    if int(input_ids.shape[1]) <= 0:
        raise ValueError("prompt produced no input tokens")

    lm_head = model.get_output_embeddings()
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise RuntimeError("model output embeddings are unavailable")
    lm_weight = lm_head.weight

    rows: List[Dict[str, Any]] = []
    effective_topk = min(topk, int(lm_weight.shape[0]))
    eos_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        logits_next = out.logits[0, -1, :]
        log_probs_next = torch.log_softmax(logits_next, dim=0)
        next_token_id = int(torch.argmax(logits_next).item())
        baseline_logprob = float(log_probs_next[next_token_id].item())

        top_idx, top_probs, baseline_entropy = topk_probs_and_entropy(
            logits_next, topk=effective_topk
        )

        hidden_last_layer = out.hidden_states[-1][0]  # [seq, hidden]
        v_raw = hidden_last_layer[-1]

        seq_len = int(hidden_last_layer.shape[0])
        if seq_len <= 1:
            splus_raw = torch.zeros(
                int(hidden_last_layer.shape[1]),
                device=hidden_last_layer.device,
                dtype=hidden_last_layer.dtype,
            )
        else:
            attn_to_past = out.attentions[-1][0, :, -1, :-1]  # [heads, seq-1]
            hidden_past = hidden_last_layer[:-1, :]
            splus_raw = compute_splus_from_past(
                attn_to_past=attn_to_past,
                hidden_past=hidden_past,
                hidden_dim=int(hidden_last_layer.shape[1]),
            )

        selected_rows = lm_weight[top_idx]
        sminus_raw = torch.matmul(
            top_probs.to(dtype=selected_rows.dtype),
            selected_rows,
        )

        v_8d = project_fwht_to_8(v_raw.detach().cpu().tolist())
        splus_8d = project_fwht_to_8(splus_raw.detach().cpu().tolist())
        sminus_8d = project_fwht_to_8(sminus_raw.detach().cpu().tolist())

        row = {
            "step": step,
            "absolute_pos": int(input_ids.shape[1] - 1),
            "token_id": next_token_id,
            "token_str": tokenizer.convert_ids_to_tokens(next_token_id),
            "V_8d": v_8d,
            "Splus_8d": splus_8d,
            "Sminus_8d": sminus_8d,
            "baseline_logprob": baseline_logprob,
            "baseline_entropy": baseline_entropy,
        }
        rows.append(row)

        next_token = torch.tensor([[next_token_id]], dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_id is not None and next_token_id == int(eos_id):
            break
    details = {
        "mode": "autoregressive_generate_v1",
        "target_token_count_expected": None,
        "target_token_count_extracted": len(rows),
        "exact_token_match_ratio": None,
        "alignment_method": None,
        "bos_prepended_for_teacher_forcing": False,
    }
    return rows, {
        "topk_effective": effective_topk,
        "mode_details": details,
    }


def run_teacher_forcing_extraction(
    prompt: str,
    target_answer: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    topk: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Teacher forcing requires a fast tokenizer (offset mapping support)."
        )

    full_text = prompt + target_answer
    answer_char_start = len(prompt)

    encoded_full = tokenizer(
        full_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded_full["input_ids"].to(device)
    offsets_tensor = encoded_full["offset_mapping"][0]
    offsets = [[int(x[0]), int(x[1])] for x in offsets_tensor.tolist()]

    if int(input_ids.shape[1]) <= 0:
        raise ValueError("teacher forcing full_text produced no tokens")

    target_token_indices = select_target_token_indices_from_offsets(
        offsets=offsets,
        answer_char_start=answer_char_start,
    )
    if not target_token_indices:
        raise ValueError("no target tokens selected from offset mapping")

    target_only_ids = tokenizer(
        target_answer,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    target_only_count = int(target_only_ids.shape[0])
    expected_target_count = len(target_token_indices)

    lm_head = model.get_output_embeddings()
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise RuntimeError("model output embeddings are unavailable")
    lm_weight = lm_head.weight
    effective_topk = min(topk, int(lm_weight.shape[0]))

    bos_prepended = False
    if target_token_indices[0] == 0:
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            raise RuntimeError(
                "first target token index is 0 and tokenizer has no BOS token; "
                "cannot compute logits[t-1] for teacher forcing"
            )
        bos_tensor = torch.tensor([[int(bos_id)]], dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([bos_tensor, input_ids], dim=1)
        target_token_indices = [idx + 1 for idx in target_token_indices]
        bos_prepended = True

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_last = out.hidden_states[-1][0]  # [seq, hidden]
    rows: List[Dict[str, Any]] = []

    for local_step, abs_t in enumerate(target_token_indices):
        t = int(abs_t)
        if t <= 0:
            raise RuntimeError("teacher forcing encountered t<=0 after BOS handling")

        v_raw = hidden_last[t, :]
        attn_to_past = out.attentions[-1][0, :, t, :t]  # [heads, t]
        hidden_past = hidden_last[:t, :]
        splus_raw = compute_splus_from_past(
            attn_to_past=attn_to_past,
            hidden_past=hidden_past,
            hidden_dim=int(hidden_last.shape[1]),
        )

        logits_prev = out.logits[0, t - 1, :]
        log_probs_prev = torch.log_softmax(logits_prev, dim=0)
        actual_token_id = int(input_ids[0, t].item())
        baseline_logprob = float(log_probs_prev[actual_token_id].item())

        top_idx, top_probs, baseline_entropy = topk_probs_and_entropy(
            logits_prev, topk=effective_topk
        )
        selected_rows = lm_weight[top_idx]
        sminus_raw = torch.matmul(
            top_probs.to(dtype=selected_rows.dtype),
            selected_rows,
        )

        offset_index = int(t - 1) if bos_prepended else int(t)
        tok_start, tok_end = offsets[offset_index]
        local_char_start = max(0, tok_start - answer_char_start)
        local_char_end = max(local_char_start, tok_end - answer_char_start)
        local_char_end = min(local_char_end, len(target_answer))

        row = {
            "step": int(local_step),
            "absolute_pos": int(t),
            "answer_char_start": int(local_char_start),
            "answer_char_end": int(local_char_end),
            "token_id": actual_token_id,
            "token_str": tokenizer.convert_ids_to_tokens(actual_token_id),
            "V_8d": project_fwht_to_8(v_raw.detach().cpu().tolist()),
            "Splus_8d": project_fwht_to_8(splus_raw.detach().cpu().tolist()),
            "Sminus_8d": project_fwht_to_8(sminus_raw.detach().cpu().tolist()),
            "baseline_logprob": baseline_logprob,
            "baseline_entropy": baseline_entropy,
        }
        rows.append(row)

    extracted_target_count = len(rows)
    if expected_target_count <= 0:
        exact_token_match_ratio = 0.0
    else:
        exact_token_match_ratio = extracted_target_count / float(expected_target_count)

    if exact_token_match_ratio < 0.98:
        raise RuntimeError(
            "exact_token_match_ratio below threshold: "
            f"{exact_token_match_ratio:.6f} < 0.98"
        )

    details = {
        "mode": "teacher_forcing_forward_v1",
        "target_token_count_expected": expected_target_count,
        "target_token_count_extracted": extracted_target_count,
        "exact_token_match_ratio": exact_token_match_ratio,
        "alignment_method": "offset_overlap_v1",
        "answer_char_start": answer_char_start,
        "target_token_indices_count": len(target_token_indices),
        "target_only_token_count": target_only_count,
        "boundary_merge_token_delta": extracted_target_count - target_only_count,
        "bos_prepended_for_teacher_forcing": bos_prepended,
    }
    return rows, {
        "topk_effective": effective_topk,
        "mode_details": details,
    }


def main() -> int:
    args = parse_args()
    prompt = read_prompt_text(args)
    target_answer = read_target_answer_text(args)
    prompt_sha256 = sha256_bytes(prompt.encode("utf-8"))
    target_answer_sha256 = (
        sha256_bytes(target_answer.encode("utf-8")) if target_answer is not None else None
    )

    configure_reproducibility(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    model_candidates = build_model_candidates(args.model_id)

    model_id, tokenizer, model, model_revision = load_first_available_model(
        model_candidates=model_candidates,
        device=device,
    )

    if target_answer is not None:
        result_rows, result_meta = run_teacher_forcing_extraction(
            prompt=prompt,
            target_answer=target_answer,
            model=model,
            tokenizer=tokenizer,
            device=device,
            topk=args.topk,
        )
    else:
        result_rows, result_meta = run_autoregressive_extraction(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
            topk=args.topk,
        )
    rows = result_rows
    effective_topk = int(result_meta["topk_effective"])
    mode_details = result_meta["mode_details"]

    out_path = Path(args.out)
    ndjson_sha256 = write_ndjson(out_path, rows)
    meta_path = out_path.parent / "meta.json"

    meta = {
        "model_id": model_id,
        "model_revision": model_revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "seed": args.seed,
        "topk_requested": args.topk,
        "topk_effective": effective_topk,
        "max_new_tokens": args.max_new_tokens,
        "proj_id": PROJ_ID,
        "splus_def_id": SPLUS_DEF_ID,
        "sminus_def_id": SMINUS_DEF_ID_TEMPLATE.format(topk=effective_topk),
        "prompt_sha256": prompt_sha256,
        "target_answer_sha256": target_answer_sha256,
        "output_ndjson_sha256": ndjson_sha256,
        "output_ndjson_path": str(out_path.as_posix()),
        "device": str(device),
        "dtype": "float32",
        "deterministic_requested": bool(args.deterministic),
        "n_steps_written": len(rows),
        "extraction_mode": mode_details["mode"],
        "alignment_method": mode_details.get("alignment_method"),
        "target_token_count_expected": mode_details.get("target_token_count_expected"),
        "target_token_count_extracted": mode_details.get("target_token_count_extracted"),
        "exact_token_match_ratio": mode_details.get("exact_token_match_ratio"),
        "bos_prepended_for_teacher_forcing": mode_details.get(
            "bos_prepended_for_teacher_forcing"
        ),
        "answer_char_start": mode_details.get("answer_char_start"),
        "target_token_indices_count": mode_details.get("target_token_indices_count"),
        "target_only_token_count": mode_details.get("target_only_token_count"),
        "boundary_merge_token_delta": mode_details.get("boundary_merge_token_delta"),
    }
    write_meta_json(meta_path, meta)

    print(f"model_id={model_id}")
    print(f"model_revision={model_revision}")
    print(f"extraction_mode={mode_details['mode']}")
    if mode_details.get("exact_token_match_ratio") is not None:
        print(f"exact_token_match_ratio={mode_details['exact_token_match_ratio']:.6f}")
    print(f"n_steps={len(rows)}")
    print(f"output={out_path}")
    print(f"meta={meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
