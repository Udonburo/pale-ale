# Gate12C-2 frozen exploratory implementation

This is the frozen, retained implementation of the Gate12C-2 synthetic smoke
and the consumed N1 locked calibration. It implements the Annex A residual
decomposition directly and does not import any retired Gate12C-2 module.

Final program state: `LOCKED_FAIL / CLOSED / REAL_NOT_AUTHORIZED`. The code is
retained for audit, tests, and synthetic exploration. It is not an active
prospective runner and does not authorize another locked attempt.

The frozen smoke interface remains available as development/audit tooling:

```powershell
python -m tools.gate12c2_minimal.run `
  --spec tools/gate12c2_minimal/study.json `
  --output runs/gate12c2-minimal-v0.1

python -m tools.gate12c2_minimal.validate `
  --spec tools/gate12c2_minimal/study.json `
  --output runs/gate12c2-minimal-v0.1
```

If a process stops after writing some shards, rerun the first command with
`--resume`. Existing shards are validated before they are reused. A completed
run cannot be overwritten or resumed.

The output contains only:

```text
study.json
state.json
shards/<case>__<regime>.json
result.json
manifest.json
```

The `SMOKE_PASS` meant that the numerical identities, N1 realizability, and the
declared S0/S1/S2 development controls completed on all 24 endpoints. It did
not itself authorize a locked test, held-out run, or scientific claim.

## Frozen smoke: 2026-08-12

The frozen v0.1 smoke study completed twice with byte-identical outputs:

```text
decision:        SMOKE_PASS
shards:          36
S0 support:      0 / 24 endpoints
S1 direction:   24 / 24 endpoints
S2 inflation:   24 / 24 endpoints
study SHA-256:   1247d20b20931f32dc23a52d564affa453e393da1994a96f8790022ba21912d9
result SHA-256:  37d78b454d851a0c0c08cb3988b9604c00f419595f59c5d83d33b6e61fdfb6aa
manifest SHA-256:f57cf5dc4737b95ec649139acd61406b4af28adae3f73525738c26b8c06a0df6
```

## Consumed locked synthetic calibration: 2026-08-12

`locked_calibration.json` is the frozen, already-consumed specification for one
separately seeded N1 calibration. One shard was one case within one independent
synthetic dataset; the dataset, not an inner reassignment, was the inference
unit. The run persisted all Annex A components and gauge-transformed components.

```text
decision:         RETIRE_OR_DEMOTE
shards:           2,880
passing gates:    mechanical, S0 familywise, S0 promotion, S1 primary, S2, nuisance
failing gate:     stability p95 shift 0.306184 > 0.30
study SHA-256:    f8586e10fece74472a7d327f6fd13a9ed0b6d813c5582873443bf01936e00262
analysis SHA-256: 15ca5bd45d3aaf79f41464ff6c8ff0bd7a555a4897b07e7ea3ec0e4bf2caea49
manifest SHA-256: de439fb4b92db1618f7a9c96383e2012b0fb3d3b47b06c9d9100474c3b4b7f8c
```

The retained result can be independently revalidated without importing the
primary experiment, analysis, or runner modules:

```powershell
python -m tools.gate12c2_minimal.locked_validate `
  --spec tools/gate12c2_minimal/locked_calibration.json `
  --output <retained-output-directory>
```

The locked runner is retained for provenance and tests. Its completed attempt
cannot be resumed or overwritten, and it must not be used to open a replacement
suite. N1 is retired as a prospective primary-null track; real held-out,
replacement locked, N2, and N3 work remain unauthorized.
