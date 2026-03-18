# Gate6 Standing And Outcome

Status: Draft
Role: Tracked standing snapshot / implementation-facing closeout
Date: 2026-03-18

## 0. Scope

This document records the current empirical standing of the Gate6 observation-redesign workstream.

It is not a theorem claim.
It is a repo-level decision snapshot based on the current full matched artifacts and helper validation.

## 1. Executive Read

Gate6 has already produced a real mainline advance.

What is now established:

- Gate6-A compatibility-only reruns were a clean negative result
- object-native consumers are not dead ends
- `F` is not a mandatory core signal in the post-FWHT line
- the current repo should explicitly carry two layers:
  - an operational candidate
  - a research north star

The current two-layer reading is:

- operational candidate: `gate6f`
- research north star: `gate6h`

## 2. Closed Results

### 2.1 Gate6-A compatibility lane

Question:

- does `same motif + local8 compatibility` beat the old fixed triad-loop benchmark?

Answer:

- no

Standing:

- builder validity: yes
- provenance packaging: yes
- promotion: no

Interpretation:

- the native local object is valid
- the `compat_local8` bridge is not the breakthrough point

Primary references:

- [`runs/gate5_cfa_gate6a_v0/gate5_aggregate_report.md`](runs/gate5_cfa_gate6a_v0/gate5_aggregate_report.md)
- [`runs/gate5_seam_gate6a_v0/gate5_aggregate_report.md`](runs/gate5_seam_gate6a_v0/gate5_aggregate_report.md)
- [`runs/gate6_cfa_boundary_standing.md`](runs/gate6_cfa_boundary_standing.md)
- [`runs/gate6_seam_boundary_standing.md`](runs/gate6_seam_boundary_standing.md)

Decision:

- `Gate6-A = kill as promotion path`

### 2.2 Object-native early consumer lines

Edge-plane holonomy:

- useful as an object-native smoke
- not strong enough on Seam quietness

Ray-projector line:

- `v1` became the first real keep candidate
- `v2` was a clean negative narrow iterate

Decision:

- edge-plane line: closed
- ray-projector line: informative, but no longer current best

Primary references:

- [`runs/gate6b_cfa_full/gate6b_aggregate_summary.md`](runs/gate6b_cfa_full/gate6b_aggregate_summary.md)
- [`runs/gate6b_seam_pairs_full/gate6b_seam_report.md`](runs/gate6b_seam_pairs_full/gate6b_seam_report.md)
- [`runs/gate6c_cfa_full/gate6c_aggregate_summary.md`](runs/gate6c_cfa_full/gate6c_aggregate_summary.md)
- [`runs/gate6c_seam_pairs_full/gate6c_seam_report.md`](runs/gate6c_seam_pairs_full/gate6c_seam_report.md)
- [`runs/gate6d_cfa_full/gate6d_aggregate_summary.md`](runs/gate6d_cfa_full/gate6d_aggregate_summary.md)
- [`runs/gate6d_seam_pairs_full/gate6d_seam_report.md`](runs/gate6d_seam_pairs_full/gate6d_seam_report.md)

## 3. Current Winners

### 3.1 Operational candidate: `gate6f`

Method:

- `sigma_gap_tailkeep_weighted_gram_loop_v2`
- current `F` reweighted by object-native singular-spectrum structure

Why it matters:

- strongest current operational balance between CFA localization and Seam quietness

Headline artifacts:

- [`runs/gate6f_cfa_full/gate6f_aggregate_summary.md`](runs/gate6f_cfa_full/gate6f_aggregate_summary.md)
- [`runs/gate6f_seam_pairs_full/gate6f_seam_report.md`](runs/gate6f_seam_pairs_full/gate6f_seam_report.md)

Current read:

- best keep candidate for operational mainline
- strong bridge between the old `F` readout and the new object-native observables

### 3.2 Research north star: `gate6h`

Method:

- `sigma_sqrtgap_tailkeep_object_v2`
- pure object-native singular-spectrum signal
- no multiplication by `F`

Why it matters:

- demonstrates that a pure object-native signal can stand on CFA without relying on `F`

Headline artifacts:

- [`runs/gate6h_cfa_full/gate6h_aggregate_summary.md`](runs/gate6h_cfa_full/gate6h_aggregate_summary.md)
- [`runs/gate6h_seam_pairs_full/gate6h_seam_report.md`](runs/gate6h_seam_pairs_full/gate6h_seam_report.md)

Current read:

- strongest pure object-native candidate so far
- not yet the operational winner on quietness
- already the architectural north star

## 4. Standing Table

| line | CFA role | Seam role | current decision |
|---|---|---|---|
| `gate6a_v0` local8 compatibility | clean negative | clean negative | kill |
| `gate6b` edge-plane holonomy | informative | weak quietness | kill |
| `gate6c` ray-projector v1 | first keep | partial quietness win | superseded |
| `gate6d` ray-projector v2 | clean negative iterate | clean negative iterate | kill |
| `gate6e` sigma-gap bridge v1 | strong | strong | keep |
| `gate6f` sigma-gap tailkeep bridge v2 | strongest bridge | strongest operational quietness balance | operational candidate |
| `gate6g` pure object v1 | first serious pure-object keep | strong | keep |
| `gate6h` pure object v2 | strongest pure-object CFA | strong, but still slightly behind `gate6f` on `mean_delta_max` | research north star |

## 5. Architecture Rule Going Forward

The repo now explicitly carries two distinct statuses for post-Gate6 consumers:

- `operational candidate`
  - best current balance for practical matched-slice use
- `research north star`
  - best current pure object-native direction even if it is not yet the operational quietness winner

These statuses are allowed to diverge.
They must not be collapsed into a single winner by rhetoric alone.

At the current snapshot:

- operational candidate = `gate6f`
- research north star = `gate6h`

## 6. What Is Left In Gate6

Very little is left.

Gate6 is effectively complete once the following are accepted as fixed:

- `Gate6-A` compatibility lane closed as a clean negative
- `gate6f` retained as operational best bridge
- `gate6h` retained as pure object-native north star

No further Gate6 score-family proliferation is recommended by default.

## 7. Handoff To The Next Workstream

The next workstream should not start by widening the score zoo.

The next minimal step is:

- projector-based progression leakage from `P_t` to `V_{t+1}`

That means the correct first downstream dynamic motif is:

- object-native progression transport
- not benchmark policy
- not retrieval conflict
- not persistent topology yet

The next workstream should therefore begin with a minimal progression-leak consumer and evaluate it on the same CFA / Seam surfaces before adding field aggregation.

That first dynamic smoke is now recorded separately in:

- [`11_GATE7_PROGRESSION_LEAK_SMOKE.md`](11_GATE7_PROGRESSION_LEAK_SMOKE.md)

Its current reading is:

- real signal: yes
- promotion: no
- field aggregation unlock: not yet
