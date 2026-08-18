# Graph-XOR R1.2-8B — Fresh Formation-Boundary Scout

STATUS: FRESH_8B_ONLY_SCOUT / EXECUTION_AUTHORIZED  
PARENT: R1.2 remains `OPERATIONAL_ABORT_OA1_FINAL` and immutable.  
RUNNER_SHA256: 51a7d8e78bad69dd9980152a9c50ccdfe4568203553163a5d3f760487b8a182d

## Question

Does Qwen3-8B, under the same frozen input-output-only ICL surface, acquire
length-8 parity from demonstrations after the 4B model acquired only the
two-input XOR mapping?

## Frozen scope

- Model: `Qwen/Qwen3-8B`
- Revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Runtime: L40S, BF16, PyTorch 2.7.1+cu126, Transformers 5.15.0
- Official non-thinking chat template; zero generated tokens; raw forced-choice
  next-token logits for contextual single-token literals `0` and `1`
- No activation extraction, CoT, scratchpad, quantization, model comparison,
  P5, A2-M, B, or semantic transfer

## Prospective design

- Fresh seed: `202608172137`
- Fresh prompt/case namespace and case ledger; no reuse of the 24 unpersisted
  8B attempts from R1.2
- Surfaces: `ICL-P2` and `ICL-P3` only
- Shots: `4`, `16`, `64`
- Conditions: correct demonstrations and blockwise-balanced label-shuffled
  demonstrations
- Targets: 24 per cell, 12 exact answer-flip matched pairs
- Total: `2 × 3 × 2 × 24 = 288` scored forwards; hard ceiling `300`
- Both surfaces are run once. P2 does not gate opening P3.

`FORMATION_PASS` is unchanged from R1.2: the correct-demonstration cell must
be `BEHAVIOR_PASS`, the shuffled control must not be, and correct-minus-control
accuracy and paired-direction deltas must each be at least `0.125`.

## Interpretation and stop

- P3 `FORMATION_PASS`: 8B exhibits bounded behavioral acquisition of length-8
  parity under this ICL protocol. This does not establish an internal XOR
  algorithm, obstruction representation, naturality, or causal use.
- Otherwise: close the bounded Qwen3-8B input-output-only ICL parity line.
- Any interruption is an operational abort; there is no resume or retry.
- No later model, surface, metric, or threshold may be added to this scout.

## Exact binding

TOKEN_BINDING_STATUS: TOKENIZER_ONLY_COMPILE_PASS  
TOKEN_BINDING_SHA256: ef1c17c4dc82cbaa303b3963b759e8593cd1050a3126749467d99092bde2de19  
CASE_LEDGER_SHA256: 36c03f6b3ee289866cfbfc9524803541294cbf73b75494945ba7c6058606249d  
MAXIMUM_PREFIX_TOKEN_POSITIONS: 4795  
CONTEXT_MARGIN_POSITIONS: 36164  
DIRECT_LITERAL_TOKEN_IDS: 0=15, 1=16
