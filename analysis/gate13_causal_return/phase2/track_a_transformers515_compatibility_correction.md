# Track A Transformers 5.15 compatibility correction

`SCIENTIFIC_RESULT = NONE`  
`CORRECTION_CLASS = INTERFACE_COMPATIBILITY_ONLY`

This record binds the one-time correction authorized after execution
`aca4b3c1-de70-4d75-a715-16d351f3f6da` stopped at
`SCIENTIFIC_RUNNER_BLOCKER`. That attempt remains immutable: it consumed one
model forward, produced zero model responses, and left A0/A1/A2 unopened. Its
blocker artifact SHA-256 is
`99e818622ff92f3955ff5cd1654551c80502accdd4644059ba5ae408880a0b0f`.

## Exact compatibility diff

Only `tools/gate13_causal_return/track_a/phase2_runner.py::_generate_one`
changed. The runner SHA-256 moves from
`2c1401ac0077aafe3d4e14a8636c2c24ef6d8ecc8f52d6abe8ee072d5961d9f8`
to
`a41dfbdba970afdbf91fc3d7bfd1d13b1acfd2d875e10b69218019fcac08e9c4`.

The old positional call was:

```python
output = model.generate(
    inputs,
    do_sample=False,
    max_new_tokens=max_tokens,
    pad_token_id=tokenizer.eos_token_id,
)
```

The corrected interface constructs `model_inputs = dict(encoded)`, rejects
any collision with the unchanged generation kwargs before a forward, and calls:

```python
output = model.generate(**model_inputs, **generation_kwargs)
```

The continuation boundary now reads the same input tensor through
`model_inputs["input_ids"].shape[-1]`. Prompt rendering, tokenizer arguments,
device placement, input tensor objects, generation strategy and values, decode
path, parser, oracle, metrics, thresholds, cases, stopping rules, model, and
runtime binding are unchanged.

## Regression boundary

Local tests pass for keyword-only dispatch, tensor object/shape/dtype/device/
content identity, fail-closed kwargs collision, and the unchanged prompt,
template, and decode contract. The exact-package CPU Modal regression also
passes in Python 3.11.2, torch 2.7.1+cu126, Transformers 5.15.0, and tokenizers
0.22.2. A randomly initialized tiny local Qwen3 model completed
`model.generate(**batch_encoding, max_new_tokens=1)` through GenerationMixin;
the input tensors were unchanged. No scientific weights, scientific case
output, GPU, model download, or scientific forward was used. The ignored local
receipt SHA-256 is
`e99c3864831a079ba5beef42b7a36a43e93f8bb48dd92a0de35a4ccb98e6e263`.

For the first M1 case `a0-l12-y0-early-r0-S`, the frozen manifest provides only
the raw prompt SHA-256
`3eadff30ee28a1c3ac08e7ea8817b25393aa581de9069a7d1c2af1b8c2b8df7a`.
It does not provide prior hashes for the rendered prompt, `input_ids`,
`attention_mask`, or generation kwargs. Those comparisons are therefore
explicit authority gaps; the new observed hashes will not be represented as
historical matches. Forward-zero preparation observed:

- rendered prompt: `f44090ad5ecb0a8f99380aef285bf3d573aacd351f3e6374534a3a41a57477e0`
- `input_ids`: `cb1980990e91f8bbe71037b67b390718f98a4ee60c84b99cc8f9c7aadc5f1064`
- `attention_mask`: `feb75c4d5648e76e10adc86194238b6eaeaaa52036a3f79799a677cd6353447e`
- generation kwargs: `24f7000ea348e9202091e5429d81b6a20f8da588452ffb64220944ebe32b8bc3`

The exact model Volume was rehashed as the same 15-file,
16,397,461,266-byte directory identity
`935acddbddad11307d52408e0e0125d4fe328ed2ab9e3af2d4bc18299f5ece14`;
no download was attempted.

The cumulative prior forward count is 1. The remaining frozen-program ceiling
is 599 of 600. A fresh execution requires a separate v2 authorization, fresh
execution identity, fresh Modal image, and fresh result Volume. No automatic
second compatibility repair is authorized.
