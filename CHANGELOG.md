# Changelog

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
