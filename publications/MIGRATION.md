# Publication Layout Migration

On 31 August 2026, the six tracked publication packages were consolidated
under `publications/<publication-slug>/zenodo/`. This was a repository-layout
change only. No file inside a published package was edited.

The content-tree SHA-256 below is computed from the UTF-8 serialization of
sorted lines in the form `<relative-path>\t<file-sha256>\n`. Counts, total
bytes, and tree identities matched before and after relocation.

| Historical path | Current path | Files | Bytes | Content-tree SHA-256 |
| --- | --- | ---: | ---: | --- |
| `zenodo-release/` | `publications/structural-replay-fp32/zenodo/` | 7 | 328,502 | `cf1f384cd3b7c0330218e0e335578222a016db05016bb957c8409a7cc94ee081` |
| `zenodo-release-transport-first-defect-telemetry/` | `publications/transport-first-defect-telemetry/zenodo/` | 5 | 235,637 | `c926b923dfb341e4399c849b35a7cc32f3856c831862d3e527e8f779c971ce13` |
| `zenodo-release-gate12b-observer-relative-closure-signatures/` | `publications/observer-relative-closure-signatures/zenodo/` | 8 | 336,722 | `aea3d7ff4e9645e3d9b3479e5a7602c610e41febab638bf6d45b917970081b6b` |
| `zenodo/gate12c1-parenthesization-defects/` | `publications/compression-interleaved-parenthesization-defects/zenodo/` | 55 | 7,690,015 | `0e30ab779801ecf60232d063faf96f6e2b4a151b1f6de7255cb465d499ef75b3` |
| `zenodo/local-mapping-without-iterative-closure/` | `publications/local-mapping-without-iterative-closure/zenodo/` | 28 | 1,179,041 | `e9f79bcabbbf489625be83fad64b01f5a597a84acf25cae371c060a4f667af9b` |
| `zenodo/sensitivity-without-reproducibility/` | `publications/sensitivity-without-reproducibility/zenodo/` | 28 | 573,565 | `8f072a7337beb7d21217165a014528f8aca9937126709700d666e8c60e411c42` |

The checksum inventories inside all six packages were also re-evaluated after
the move: 91 declared file hashes passed. Existing Zenodo records, DOI
metadata, GitHub Release tags, and assets were not modified.
