# pale-ale Trace Triage Demo

Static first-contact demo for pale-ale Trace Triage.

It is designed for a busy evaluation, red-team, agent-monitoring, AI safety, or
grant reviewer who needs to understand the practical output in about 30 seconds:

> Scalar-only evaluation says pass. pale-ale shows the 3 trace rows a human should inspect first.

![Trace Triage demo first screen](screenshots/trace-triage-demo-hero-chrome.png)

[Full-page screenshot](screenshots/trace-triage-demo-fullpage-chrome.png)

## Live Demo

Live demo:

https://pale-ale-trace-triage.vercel.app/

Local development URL, when the dev server is running:

`http://127.0.0.1:5173/`

## What This Shows

The demo uses a synthetic policy/RAG/evaluation trace where the final scalar
checks pass, but a source constraint changes inside the trace.

The pale-ale-facing output is a shortlist of rows for human review:

- Row 03: retrieved context omitted an exclusion.
- Row 04: a requirement was broadened.
- Row 06: the final answer conflicts with the source.

This is a triage surface. It tells a reviewer where to inspect first.

## Why It Exists

The demo is a compact external-validation artifact. It is meant to support
feedback conversations with evaluation, monitoring, red-team, AI safety, grant,
or pilot collaborators who want to know whether artifact-row triage could be
useful for their long LLM or agent traces.

It is built from:

- `../../docs/demo/trace-triage/storyboard.md`
- `../../docs/demo/trace-triage/synthetic-trace.json`

It is a separate static app. It does not modify or replace RANVIER.

## Run Locally

```powershell
cd apps/trace-triage-demo
npm install
npm run dev
```

Vite will print a local URL, usually `http://localhost:5173/`.

## Build

```powershell
cd apps/trace-triage-demo
npm run build
```

The production bundle is written to `apps/trace-triage-demo/dist/`, which is ignored by git.

## Deployment Notes

The app is deployed on Vercel at:

https://pale-ale-trace-triage.vercel.app/

The app is static after build. Suitable deployment targets include Vercel,
Netlify, or GitHub Pages.

Suggested settings:

- project name: `pale-ale-trace-triage`
- root directory: `apps/trace-triage-demo`
- install command: `npm ci`
- build command: `npm run build`
- output directory: `dist`

## Limitations

- Static fixture data only.
- No backend.
- No model calls.
- No external API calls.
- Synthetic illustrative trace, not a benchmark result.
- The same-review-budget comparison uses fixed illustrative selections from the storyboard.
- Its counted metric is first-inspect targets included, not every row that may be useful as supporting context.

## Claim Boundaries

- Not a benchmark.
- Not a correctness classifier.
- Not a model-quality score.
- Not a deception detector.
- Not a claim about model internals.
- Synthetic illustrative example; bounded evidence linked separately.

## Relationship to RANVIER

RANVIER remains a document-grounded constraint audit sidecar.

pale-ale Trace Triage is separate. It is for long LLM, agent, RAG, or evaluation trace human-review prioritization.

## Related Links

- [GitHub repository](https://github.com/Udonburo/pale-ale)
- [Demo source docs](../../docs/demo/trace-triage/README.md)
- [Outreach memo](../../docs/outreach/common-memo-v0.4.2-validation.md)
- [Gate12A frozen technical report](https://doi.org/10.5281/zenodo.19483162)
- [Transport-first telemetry note](https://doi.org/10.5281/zenodo.19569052)
- [Gate12B observer-relative closure signatures](https://doi.org/10.5281/zenodo.20080003)
