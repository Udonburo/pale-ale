# Graph-XOR R1.1 — Capability Localization Scout

```text
STATUS: PLAN_FROZEN / ONE_SHOT_EXECUTION_AUTHORIZED
R1 v1.0: CLOSED_AT_C0 / S0_SELECTION_FAIL / IMMUTABLE
R1.1: separate one-shot behavioral scout; not an amendment or rescue
```

## Question and boundary

At what earliest stage—codebook readout, relation interpretation, parity composition,
relational presentation, serialization/order dependence, or rank-2 integration—does
zero-shot behavioral capability become detectable in the Qwen3 size family under the
official non-thinking, no-demonstration, single-forward forced-choice surface?

R1.1 does **not** inspect activations, claim latent absence, test thinking mode/ICL/fine-
tuning, or estimate a causal parameter-count effect. A failure closes only the Qwen3
0.6B–8B zero-shot/non-thinking/direct-semantic graph-XOR path. R1 v1.0 remains bound to
its closed scientific specification and manifest; neither is edited.

## Frozen surfaces, cases, and opening rule

| Surface | Cases | Exact role |
|---|---:|---|
| I0 | 64 | explicit semantic bit → balanced randomized codebook |
| P0 | 60 | explicit semantic-bit echo |
| P1 | 60 | one-edge relation bit |
| P2 | 60 | two-edge XOR composition |
| P3 | 60 | length-8 ordered raw-bit parity |
| P4 | 60 | the same 30 matched bit problems as ordered relational cycles |
| P5 | 60 | the same P4 lines with only edge serialization shuffled |
| X0 | 64 | shuffled relational cycle + balanced randomized codebook |
| A2-M | 60 conditional | 30 exact surface-matched unicyclic pairs |
| B | 60 conditional | 10 decorated-theta morphologies × 3 alpha × 2 answers |

Every model receives I0, P0–P5, and X0. P5 `BEHAVIOR_PASS` opens direct A2-M; A2-M
`BEHAVIOR_PASS` opens direct B. Models open in the exact order 0.6B → 1.7B → 4B → 8B.
The first direct-B `BEHAVIOR_PASS` stops the entire scout and leaves larger models
unopened. Failure through 8B closes this path. `SCORE_SIGNAL_ONLY` never opens anything.

P3→P4 is a **composite relational-presentation burden** (language, entity binding,
length, and serialization), not a pure causal estimate. P4/P5 must have equal token
counts and token-ID multisets; only constraint-line order may differ. No padding control
is introduced. Maximum scored cases are 2,432; warm-up/timing forwards are zero; the
absolute forward ceiling is 2,500.

## Scores, classifications, and fail-safe diagnostics

Direct score: `s = logit(semantic 1) - logit(semantic 0)`. Codebook score uses the
physical token currently bound to semantic 1 minus the token bound to semantic 0.
Raw logits and raw forced-choice behavior remain primary; no dummy-prompt subtraction
or recentered accuracy may authorize progression. Exact score ties earn 0.5 accuracy;
the logged categorical choice uses the lower token ID.

For matched pairs, `D_pair = mean[1(s1>s0) + 0.5*1(s1=s0)]`. AUC uses the same semantic
score with 0.5 credit for ties. With `mu_y = E[s|y]`, descriptive readout offset and
separation are `(mu_1+mu_0)/2` and `(mu_1-mu_0)/2`; neither alters classification.

```text
BEHAVIOR_PASS:
  accuracy >= .75 AND D_pair >= .75
  AND every applicable frozen stratum accuracy >= .625
  AND all collapse flags are false

SCORE_SIGNAL_ONLY:
  not BEHAVIOR_PASS AND (AUC >= .75 OR D_pair >= .75)

NO_DETECTED_SIGNAL:
  neither condition
```

`semantic_label_collapse := max_y n(pred=y)/N >= .95`.
For candidate pair `p`, `physical_token_collapse := max_p max_{t in p} n_{p,t}/N_p >= .95`.
For I0/X0, `entry_position_collapse := max(n_first,n_second)/N >= .95`.
`systematic_inversion_collapse := (AUC <= .10 AND D_pair <= .10)`; a single low metric
is logged as a warning. Applicable strata are semantic answer for every surface; alpha
for B; and mapping, entry order, and token-pair identity for I0/X0. These are fail-safe
sanity checks, not additional inferential claims.

## Exact scientific and runtime bindings

Case-generation seed: `202608170931`; parity length: `8`; A2-M profile: `(m=8,t=8)`;
theta profile: `path_internal_count=3,max_gadget_nodes=3`. Model and tokenizer commits:

```text
Qwen/Qwen3-0.6B  c1899de289a04d12100db370d81485cdf75e47ca
Qwen/Qwen3-1.7B  70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
Qwen/Qwen3-4B    1cfa9a7208912126459214e8b04321603b3df60c
Qwen/Qwen3-8B    b968826d9c46dd6066d109eabc6255188de91218
```

Tokenizer-only compile binding (PyTorch absent; 608 cases × four exact revisions):

```text
TOKENIZER_BINDING: TOKENIZER_ONLY_COMPILE_PASS
compile aggregate: 8b8290c5adcb8b906bf0f87106d93aeaf17f8c7420d65ab801b7ac830c87e968
case ledger:       bb3b2ed439ff304628978f8632203809abfc31bd5d88a7623f5542156283469e
chat template:     a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8
prefix aggregate:  52b46925a927ef6c87c2c6cd27c8ca7e79f6c0481b52eefd2e1e53516993afcd
maximum prefix:    541 tokens
direct literals:   "0" -> 15; "1" -> 16
codebook literals: " A" 362; " B" 425; " K" 730; " M" 386;
                   " R" 431; " V" 647; " X" 1599; " Z" 1863
P4/P5:             exact token-count and token-ID-multiset match PASS
common tokenizer files:
  merges.txt           8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5
  tokenizer.json       aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
  tokenizer_config     d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101
  vocab.json           ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910
per-model config.json:
  0.6B 660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd
  1.7B 1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197
  4B   8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a
  8B   f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30
RUNNER_SHA256: 6034b778e50926aeedf377e8402f0e571614d6985ffef924cd89095b65454324
```

Runtime request: one Modal `L40S` (48 GiB class), one Sandbox, at most 7,200 seconds,
no retry/replacement campaign, BF16 weights and computation, no quantization, eager
attention, TF32 disabled, `eval()` and `torch.inference_mode()`. Exact image lock:
Python 3.11.2 bullseye digest `2f749ef90f54fd4b3c77cde78eec23ab5b8199d9ac84e4ced6ae523ef223ef7b`;
PyTorch `2.7.1+cu126`; Transformers `5.15.0`; tokenizers `0.22.2`;
huggingface_hub `1.27.0`; jinja2 `3.1.6`; accelerate `1.14.0`;
safetensors `0.8.0`; Pillow `11.3.0`. The actual host driver and exact reported GPU
identity are checked against the L40S/≥40 GiB contract and recorded, not post-selected.

The pinned tokenizer chat template is applied with `add_generation_prompt=true` and
`enable_thinking=false`. Scoring uses raw next-token logits after the exact assistant
prefix (including the fixed empty-think block); `generate()`, sampling, logits processors,
generated tokens, hidden states, attentions, and hooks are forbidden. R1.1 is uniformly
BF16 and is not causally pooled or compared with the historical FP32 v1.0 result.

## Outputs and change boundary

Only four R1.1 artifacts exist: this plan, `run_r1_1.py`, `r1_1_results.json`, and
`r1_1_capability_matrix.png`. The JSON binds plan/runner hashes, commits, tokenizer and
runtime inventories, GPU identity, seed, opened/unopened models and surfaces, every
metric/stratum, forward count, and stop reason. A passing scout selects a surface only;
all later mechanism work uses fresh worlds. After final binding, changes are allowed only
for an execution-breaking implementation contradiction, tokenization leakage, or actual
compute impossibility; no new model, surface, threshold, control, or rescue is permitted.
