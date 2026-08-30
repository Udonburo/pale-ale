# Graph-XOR R1.2 — In-Context XOR Formation Scout

READ FIRST. This file is the sole plan for a bounded post-closeout scout. It does not
amend or reopen R1 v1.0 or R1.1.

```text
R1 v1.0: CLOSED_AT_C0 / S0_SELECTION_FAIL / IMMUTABLE
R1.1:    SCOUT_CLOSE_NO_DIRECT_B_THROUGH_8B / IMMUTABLE
R1.2:    ICL_FORMATION_SCOUT / EXECUTION_AUTHORIZED_AFTER_EXACT_BINDING
```

## Question and claim boundary

> Can a frozen Qwen3 model form and behaviorally use an exact XOR-composition
> variable from input-output demonstrations supplied in context, despite lacking
> zero-shot capability on the same task family?

This scout tests Qwen3-4B and, only if needed, Qwen3-8B in official non-thinking
mode with frozen weights, direct semantic 0/1 answers, no generated reasoning, no
scratchpad, no codebook, and no activation extraction. Failure is limited to this
bounded input-output-only ICL surface. It does not establish absence under thinking
mode, training, another model family, another readout, or inside hidden states.

R1.1 remains the historical zero-shot reference and is not pooled with R1.2. Any
difference cannot be attributed uniquely to demonstrations, prompt wording, BF16,
backend, or another changed execution detail.

## Surfaces and sequential opening

| Surface | Target cases | Exact task |
|---|---:|---|
| ICL-P2 | 24 | two-edge XOR composition |
| ICL-P3 | 24 | length-8 ordered raw-bit parity |
| ICL-P5 | 24 | length-8 shuffled relational cycle |
| ICL-B | 24 | direct-semantic decorated-theta cycle query |

Each surface has a fixed 64-example demonstration bank. The 4-, 16-, and 64-shot
conditions are nested prefixes of that bank. Every shot count is evaluated with:

1. `correct`: exact input-output labels;
2. `label_shuffled`: the identical demonstration inputs and order with a blockwise
   balanced output permutation.

The shuffled control preserves the output-label count in every four-example block,
changes exactly half the labels, is neither the correct sequence nor its complement,
and therefore preserves prompt length and token multiset while destroying the exact
input-output mapping.

Target instances, target order, and target labels are identical across shot counts
and controls. Demonstration and target rendered instances are disjoint. For P3/P5,
exact eight-bit patterns are also disjoint; P3 and P5 share the same underlying
matched bit problems. For B, demonstration and target decorated-theta world hashes
are disjoint. P2 has only four truth-table assignments, so assignments necessarily
recur; its held-out unit is the rendered relational instance, not a novel truth-table
row.

Models open in this order:

```text
Qwen/Qwen3-4B
Qwen/Qwen3-8B, only if 4B does not pass ICL-B
```

Within each model:

```text
ICL-P2 -> ICL-P3 -> ICL-P5 -> ICL-B
```

A surface is opened only after the preceding surface has `FORMATION_PASS`. The first
model with ICL-B `FORMATION_PASS` ends the scout and leaves every larger model
unopened. If 8B fails any prerequisite or ICL-B, the bounded Qwen3 input-output-only
ICL XOR-formation line closes. No larger model, other family, CoT, fine-tuning, or
activation analysis is automatically authorized.

## Prompt contract

Demonstrations are plain input-output pairs inside one user message. They contain no
rationale, intermediate state, or scratchpad. The target follows the demonstrations
and requests exactly one semantic digit. The exact pinned Qwen3 chat template is
applied with:

```text
add_generation_prompt = true
enable_thinking = false
```

Primary scoring uses raw next-token logits at the final position of the exact
rendered chat prefix, after the fixed empty-think block. `generate()` is forbidden.
The semantic score is:

```text
s = logit("1") - logit("0")
```

Ties receive 0.5 accuracy credit and are logged categorically by lower token ID.
No dummy-prompt or recentered accuracy may authorize progression.

## Cell metrics and formation decision

Each `(model, surface, shots, control)` cell is classified with the frozen R1.1
behavioral rule:

```text
BEHAVIOR_PASS iff
  semantic accuracy >= 0.75
  AND paired directional consistency >= 0.75
  AND every applicable frozen stratum accuracy >= 0.625
  AND no collapse flag is true

SCORE_SIGNAL_ONLY iff not BEHAVIOR_PASS
  AND (semantic-score AUC >= 0.75 OR paired consistency >= 0.75)

NO_DETECTED_SIGNAL otherwise
```

For a shot count, `FORMATION_PASS` requires all of:

```text
correct cell = BEHAVIOR_PASS
label_shuffled cell != BEHAVIOR_PASS
correct accuracy - shuffled accuracy >= 0.125
correct paired consistency - shuffled paired consistency >= 0.125
```

A surface passes if any predeclared shot count passes; the selected formation point
is the smallest passing shot count. If no shot passes, `FORMATION_SIGNAL_ONLY` is
descriptive when a correct cell is `SCORE_SIGNAL_ONLY` and exceeds its shuffled
control by at least 0.125 in AUC or paired consistency. Only `FORMATION_PASS` opens
the next surface.

Collapse detectors are exact:

```text
semantic_label_collapse := max_y n(pred=y)/N >= 0.95
physical_token_collapse := max_t n(chosen_token=t)/N >= 0.95
systematic_inversion := AUC <= 0.10 AND paired consistency <= 0.10
```

With `mu_y = E[s | y]`, `(mu_1+mu_0)/2` is descriptive score offset and
`(mu_1-mu_0)/2` is descriptive score separation. Neither changes a decision.

## Case, control, and budget contract

Each surface has 12 matched target pairs (24 cases). ICL-B uses four fresh decorated
theta morphologies x three query coordinates x two semantic answers. Labels are
balanced in every target ledger and every 4/16/64-shot demonstration prefix.

Maximum scored forwards:

```text
2 models x 4 surfaces x 3 shot counts x 2 controls x 24 targets = 1,152
absolute campaign ceiling = 1,500
```

There are no scientific retries. Preemption, timeout, runtime drift, hash drift,
tokenization mismatch, OOM, or nonzero remote exit is `OPERATIONAL_ABORT`; no second
GPU campaign is authorized by this plan.

## Exact bindings

```text
CASE_GENERATION_SEED: 202608171742
SHOT_COUNTS: [4, 16, 64]
MODEL_ORDER:
  Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c
  Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218

DIRECT_LITERALS: ["0", "1"]
DIRECT_TOKEN_IDS: [15, 16]
CHAT_TEMPLATE_SHA256: a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8
CASE_LEDGER_SHA256: 8869d95aed9f88c2f8557f5fc538eab34c8a396bb48a61bbb46f52143f1d3bc9
TOKENIZER_BINDING_SHA256: 0664a1e04fb6c9c82ab953486d54b93171b08c433934fca45ceaec4bfdccf768
PREFIX_AGGREGATE_SHA256: c9af47f32405cbc596d2e113d93a0f8919f8d967a485bed19e39f20e223ffc4a
MAX_PREFIX_TOKEN_POSITIONS: 29941
MIN_CONTEXT_MARGIN_POSITIONS: 11018
CORRECT_SHUFFLED_TOKEN_COUNT_AND_MULTISET: exact match PASS (1152/1152 prompts audited)

GPU: NVIDIA L40S, >= 40 GiB
PYTHON: 3.11.2
PYTORCH: 2.7.1+cu126
TRANSFORMERS: 5.15.0
TOKENIZERS: 0.22.2
HUGGINGFACE_HUB: 1.27.0
JINJA2: 3.1.6
ACCELERATE: 1.14.0
SAFETENSORS: 0.8.0
DTYPE: bfloat16
ATTENTION: sdpa
LOGITS_TO_KEEP: 1
TF32: false
QUANTIZATION: none
CAMPAIGN_TIMEOUT_SECONDS: 7200

RUNNER_SHA256: 1a718c8b019c055f00976fd96adbc6b326e76184c387036297d5800f948f8654
```

The runner also fail-closed verifies the immutable R1 v1.0 spec/manifest/B0 package
and the four R1.1 artifact hashes before model execution.

## Outputs and change boundary

The branch contains exactly four durable artifacts:

1. this plan;
2. `run_r1_2.py`;
3. `r1_2_results.json`;
4. `r1_2_formation_matrix.png`.

No new constitution, annex, manifest hierarchy, Rust implementation, holdout system,
activation package, or alignment code is authorized. Before execution, edits are
allowed only for an implementation contradiction, tokenizer/context failure,
compute impossibility, or immutable-binding failure. After execution, no new shot
count, surface, model, metric, or control may rescue a failed result.
