# Genealogy Supervision Policy

## Policy Status

This policy formalizes the current handling of `genealogy` after the label-geometry diagnostic workstream.

It is intentionally narrow.

## Fixed Policy

For `genealogy`:

- canonical evaluation remains `inside_span`
- canonical labels are unchanged
- the fixed Gate5 comparator remains `rotor_loop_chordal_v1`
- diagnostic geometries are supplementary views only

## Allowed Uses Of Diagnostic Geometry

Diagnostic geometry may be used to:

- explain why rotor underperforms on canonical `inside_span`
- compare onset / prefix sensitivity against canonical span labels
- motivate benchmark-side questions in future work

Diagnostic geometry may not be used to:

- replace canonical CFA labels
- restate canonical leaderboard outcomes
- claim comparator promotion
- merge canonical and diagnostic scores into one number

## Required Separation In Future Reports

If a future report includes `genealogy` diagnostics, it should separate:

- `canonical genealogy view`
- `diagnostic geometry view`

At minimum:

- canonical `inside_span`
- one diagnostic prefix/onset view

must appear in distinct tables or clearly labeled sections.

## Operational Consequence

The next workstream for `genealogy` should treat supervision / benchmark geometry as a first-class issue.

It should not default back to:

- reader proliferation
- new boundary engineering
- new residual families

unless benchmark-side evidence stops supporting the mismatch reading.
