# Sources and dependencies

The numerical generator inputs are derived from the first two coordinates of
SciPy 1.15.3's unscrambled Sobol generator, whose direction numbers use the
Joe--Kuo construction. `confirmation/data/matrices.json` under
`review_checks/primal_dual/` records the consumed direction columns and the
derived fixed matrices; Section 5.2 of the paper specifies the transformation.
They are numerical reproduction inputs, not a bundled copy of SciPy.

- S. Joe and F. Y. Kuo, *Constructing Sobol sequences with better two-dimensional
  projections*, SIAM Journal on Scientific Computing 30(5), 2635--2654 (2008).
  <https://doi.org/10.1137/070709359>
- SciPy Sobol generator documentation:
  <https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.qmc.Sobol.html>
- The Philox4x32-10 construction follows J. K. Salmon et al., *Parallel random
  numbers: as easy as 1, 2, 3* (SC 2011),
  <https://doi.org/10.1145/2063384.2063405>. The paper and source specify the
  counter addresses, domains, key and consumed words used in this experiment.

NumPy, SciPy, Numba, Matplotlib, pypandoc/Pandoc and Typst are installed
separately from the pinned requirements. Their own licenses govern those
packages. This archive does not redistribute their runtime libraries or fonts.
The repository's author-created sources use MPL-2.0; publication materials and
saved simulation results use CC BY 4.0, subject to retained third-party rights.
