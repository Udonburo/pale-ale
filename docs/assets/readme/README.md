# README artwork

The light/dark headers are illustrative artwork. The crossover charts plot all
17 saved median within-seed stream/direct ratios from the current paper's
[Table 2](../../../papers/exact-state-ranked-array-rqmc/tables/primal_dual_times.md).
Lines connect evaluated populations; they are not fitted performance models.

Regenerate from the repository root with Python and matplotlib:

```sh
python docs/assets/readme/render_assets.py
```

Use `--preview-dir PATH` for optional PNG previews. The renderer reads the
published table and writes only these documentation assets. It collects no
timings and changes no manuscript, experiment, or archived companion input.
