# Changelog

## 1.0.1 - 2026-02-16

### Fixed

- Python wheel build now consistently enables the `python-inspect` feature via `pyproject.toml`, preventing missing `PyInit_pale_ale_core` symbol issues in isolated builds.

### Added

- `scripts/build_wheel.ps1` for reproducible local wheel builds with optional install/verify flow.
- `dist/wheel_provenance.json` generation in the wheel script, including `core_git_sha` and `wheel_sha256`.

## 1.0.0 - 2026-02-03

### Changed

- **License Strategy:** Established **MPL-2.0** as the release license for v1.0.0 (Core Engine).
  - *Note: Previous pre-release artifact v0.3.0 (Apache-2.0) has been yanked due to metadata/licensing misalignment.*
- **Roadmap:** Added intent to transition to Dual License (MIT/Apache) post-whitepaper.

### Added

- Major: Initial Release.
- Core: E8 lattice snapping and geometric algebra (rotor/wedge) distance metrics.
- Safety: Full input validation (NaN/Inf rejection), no-unsafe implementation.
- Performance: Const generics optimization for k=1..3, precomputed blocks for cache locality.
