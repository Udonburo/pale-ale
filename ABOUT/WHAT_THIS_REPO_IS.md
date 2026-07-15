# What This Repo Is

`pale-ale` is a public-facing research repo for structural approaches to local observation, consistency, and learned systems.

The repo is broader than any one current method surface. It is meant to hold both:

- narrow frozen checkpoints that earn explicit public claims
- wider empirical and mathematical exploration that may later become a checkpoint, or may remain exploratory

It is best read as a public-facing experimental surface for a broader structural research program, not as the entire program in one repo-level slogan. The public-facing scope here is intentionally LLM-centered unless explicitly widened.

At its current public checkpoint, the repo is not trying to produce "one better score." It is doing something narrower and more falsifiable:

- construct local objects on a fixed observation surface
- define transport between adjacent local objects
- measure where transport fails to close cleanly
- keep those failures explicit instead of hiding them behind smoothing or aggregate scoring

## Broader Research Canvas

The broader canvas of the repo includes, but is not limited to:

- observer-relative local description and overlap structure
- gluing and consistency across local views
- trajectory-level structure and replay
- transport and closure behavior across declared surfaces
- family-conditioned versus cross-family behavior
- boundary formation, exclusions, and sidecar regimes
- relation-first boundary candidates and chart constructions
- geometry-, topology-, information-, and physics-informed mathematical imports

Not every part of that canvas is frozen into a paper claim at the same time, and not every part of that canvas needs to be stated directly in this repo's public-facing identity.

## Current Public Claim Surfaces

The frozen base paper-facing line is Gate12A under a fixed FP32
dense-transformer regime. Gate12B and Gate12C-1 are separate bounded companion
surfaces: Gate12B reads observer-relative closure signatures over existing
artifacts, while Gate12C-1 reports a predeclared parenthesization-defect null
test.

Within the Gate12A base line, the repo records:

- a fixed Gate8 to Gate12A artifact pipeline
- structural replay evidence across closed 3B/4B dense-transformer families
- narrower, exploratory phenotype summaries that are kept separate from machine-side structural pass/fail
- boundary results showing where frozen-protocol admission fails or where a sidecar does not preserve the same path signature

This is intentionally narrower than a full theory of reasoning, hallucination, or architectural universality.
It is one current testbed inside a broader research program rather than the full conceptual boundary of the repo, and it uses LLMs as the present public sandbox rather than as the final scientific boundary.

## What The Repo Is Not Claiming

The repo does not currently claim:

- a universal law for all model architectures
- a completed mechanistic account of hallucination
- a graph-wide operator success result
- a retroactive rewrite of earlier Gate8, Gate9, or Gate10 checkpoint results
- that the repo's broader exploratory canvas has already been frozen into one settled method

## Directory Map

- [`../ABOUT/`](README.md): orientation docs for human readers
- [`../workstream/`](../workstream/README.md): numbered tracked research memory
- [`../runs/`](../runs/): emitted artifacts, manifests, status files, and summaries
- [`../tools/`](../tools/): narrow Python-side runner and audit surface for the current line
- [`../src/`](../src/) and [`../crates/`](../crates/): retained code and infrastructure
- [`../docs/`](../docs/) and [`../specs/`](../specs/): supporting documentation and design surfaces

## Why The Repo Is Structured This Way

The structure is meant to separate four things that are easy to blur:

- the current scientific claim surface
- the tracked memory that explains how that surface was reached
- the emitted artifacts that support the claim
- the code that makes the artifacts and replay surfaces reproducible

The Workstream and Gate system is the repo's way of keeping those layers explicit rather than letting them collapse into one moving target. That same structure also lets the broader research canvas stay wider than the current public checkpoint without forcing every exploratory line into a premature claim.
