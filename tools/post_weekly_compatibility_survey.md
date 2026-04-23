# Post-Weekly Compatibility Survey

## Scope

This file is a read-only compatibility/admission survey for post-current weekly candidates. It does not widen the closed weekly mainline, does not create a new checkpoint, and does not make a new memo claim.

Closed weekly mainline at survey time:

- `qwen2_5_3b`
- `llama3_2_3b`
- `qwen3_4b`

## Current repo/runtime assumptions

The current model-loading path is narrow:

- `AutoTokenizer.from_pretrained(model_id, use_fast=True)`
- `AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)`
- no `trust_remote_code=True`
- no explicit auth-token plumbing in the loader path
- no `AutoProcessor` / multimodal input path in the loader path

That is a good fit for the already-closed dense text-only weekly surface. It is a poor fit for candidates that require:

- `AutoProcessor`
- multimodal prompt assembly
- `trust_remote_code`
- gated/community-license acceptance before download
- materially larger memory posture than the current single-L4 lane

Observed successful weekly-lane baseline reused for this survey:

- `transformers 5.5.4`
- `huggingface_hub 1.11.0`
- `torch 2.9.1+cu129`

## Candidate matrix

| candidate | likely supported by current stack | likely blockers | auth/gated risk | trust_remote_code risk | recommended lane |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3-4B-Instruct-2507` | yes | no clear loader delta from current `qwen3_4b`; still needs explicit admission rather than quiet widening | low | low | current mainline candidate |
| `Qwen/Qwen3-8B` | uncertain | current loader is `torch.float32`; single-L4 headroom is not obviously clean; outside current bounded 3B/4B weekly set | low | low | compatibility lane |
| `Qwen/Qwen3.6-27B` | no | official card positions this as a vision-encoder model with large serving posture; not a clean dense text-only successor | low | low | protocol-expanding lane |
| `Qwen/Qwen3.6-35B-A3B` | no | multimodal MoE plus large serving posture; far outside current weekly resource and loader assumptions | low | low | protocol-expanding lane |
| `google/gemma-4-E2B-it` | uncertain | official path uses `AutoProcessor`, Gemma-4-specific chat/response handling, and multimodal-aware prompts | low | low | compatibility lane |
| `google/gemma-4-E4B-it` | uncertain | same loader mismatch as E2B-it, plus larger artifact footprint and multimodal family defaults | low | low | compatibility lane |
| `google/gemma-4-31B-it` | no | large dense Gemma 4 plus `AutoProcessor`/multimodal path; not aligned with current single-L4 weekly lane | low | low | defer |
| `microsoft/Phi-4-mini-instruct` | no | official Transformers path uses `trust_remote_code=True` and custom-code expectations absent from the current repo loader | low | high | compatibility lane |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | no | gated/community-license path plus processor-based multimodal usage and larger resource posture | high | low | protocol-expanding lane |

## Narrow takeaways

1. The cleanest post-current weekly admission candidate is `Qwen/Qwen3-4B-Instruct-2507`.
2. The latest official `Qwen3.6` family surfaced in this survey starts at `27B` and `35B-A3B` (plus FP8 variants via the HF API), so it is not a clean next step for the current single-L4 dense weekly lane.
3. `Gemma 4` small models are plausible compatibility-lane work, but they are not clean current-mainline admissions because the repo does not currently use `AutoProcessor` or multimodal-aware prompting.
4. `Phi-4-mini-instruct` is interesting precisely because it is small enough to matter but explicitly asks for `trust_remote_code=True`; that should remain opt-in compatibility work.
5. No surveyed candidate landed in a pure auth-only lane. The gated candidates here also widen protocol or loader assumptions.

## Low-burn next moves

- If the goal is a clean dense-text successor to the current weekly line, start with `Qwen/Qwen3-4B-Instruct-2507`.
- If the goal is to open a separate compatibility lane, `google/gemma-4-E2B-it` is the smallest Gemma 4 candidate worth inspecting first.
- If the goal is specifically "latest Qwen 3.6", treat it as protocol-expanding work rather than a quiet continuation of the closed 3B/4B dense weekly set.

## Sources

Primary sources used for this survey:

- `Qwen/Qwen3-4B-Instruct-2507`: <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507>
- `Qwen/Qwen3-8B`: <https://huggingface.co/Qwen/Qwen3-8B>
- `Qwen/Qwen3.6-27B`: <https://huggingface.co/Qwen/Qwen3.6-27B>
- `Qwen/Qwen3.6-35B-A3B`: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- `Gemma 4` family docs: <https://huggingface.co/docs/transformers/model_doc/gemma4>
- `google/gemma-4-E2B-it`: <https://huggingface.co/google/gemma-4-E2B-it>
- `google/gemma-4-E4B-it`: <https://huggingface.co/google/gemma-4-E4B-it/tree/main>
- `google/gemma-4-31B-it`: <https://huggingface.co/google/gemma-4-31B-it>
- `microsoft/Phi-4-mini-instruct`: <https://huggingface.co/microsoft/Phi-4-mini-instruct>
- `meta-llama/Llama-4-Scout-17B-16E-Instruct`: <https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct>

Read-only family discovery checks were also run against the Hugging Face model API for `Qwen3.6` and `gemma-4` identifiers.
