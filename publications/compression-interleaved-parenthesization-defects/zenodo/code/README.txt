Recorded code-byte provenance
=============================

Do not normalize line endings before checking these hashes.

run_gate12c_compressed_overlap_associator.py
  Commit: 8d5613bffe5b6c91d0956c812404072eb76e98c6
  SHA-256: b363fd874a0538dc548853e97e8ec17c0eb84be5658f6e2f01f60d2a12789c3e
  Line endings: LF
  Role: measurement runner recorded by every case manifest

summarize_gate12c1_first_empirical_grid.py
  Commit: 6b66f49710f4b7cbdae7e5a282d14d1d81723b30
  SHA-256: 9e527c0363875de912f9abf9ba38acb462a13b0b62da6e14be35d4c3526df708
  Line endings: CRLF
  Role: grid summarizer recorded by results/summary/manifest.json

inspect_gate12c_associator_feasibility.py
  Commit snapshot: 8d5613bffe5b6c91d0956c812404072eb76e98c6
  Git-blob/LF SHA-256: 5b824f929f8bc145bb485b62cd5a0e409ae683fbe11e0e798ecbca123fe71f2d
  Role: shared local helper imported by both scripts

The mixed line endings are preserved because the runner was executed from the
LF-only empirical worktree, while the later grid builder recorded the CRLF
checkout bytes from its Windows checkout. This is a provenance distinction,
not a scientific-method difference.
