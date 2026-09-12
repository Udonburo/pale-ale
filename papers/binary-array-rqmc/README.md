# Exact Binary Projections for Array-RQMC

**Joint Laws and Pathwise-Preserving Execution**  
Aoi Kawasaki

Archival identifier: [doi:10.5281/zenodo.22728405](https://doi.org/10.5281/zenodo.22728405)  
Version: 1.0.0; manuscript date: 13 September 2026  
Preprint; not peer reviewed.

## Read and reproduce

main.md is the canonical manuscript. It characterizes an exact consumed joint
law, derives its support and covariance, and gives an equal-covariance nonlinear
variance witness. It then reports two execution rewrites, preservation arguments,
and the completed timing and accuracy experiments.

See [repro/README.md](repro/README.md) for the computational companion. It includes
19 correctness and saved-data tests, the three measured array kernels, saved
numerical inputs, and regeneration of the two figures and Tables 1-4. It does
not include every original raw observation, bootstrap sample, timing round, or
baseline/calibration driver. Figure regeneration is not a rerun of those studies.

The finite-law examples are mathematical calculations, not additional
performance or accuracy experiments. No private workspace, GPU, pretrained
model, or external dataset is needed for the companion.

## Archive and source identity

The version-specific Zenodo identifier above binds the manuscript and its
reproducibility capsule. The capsule's files and CHECKSUMS-SHA256.txt specify the
exact included sources. The [development repository](https://github.com/Udonburo/pale-ale)
is a browsing/development location, not a substitute for the archived files.
No package-containing Git commit or GitHub release is asserted in this deposit.

In the development repository, working sources stay under
papers/binary-array-rqmc/ and public deposit files are generated under
publications/binary-array-rqmc/zenodo/. Inside the capsule, everything lives under
one binary-array-rqmc/ root. The publication/ folder contains the deposit notes.
The DOI is reserved during preparation and becomes a public archival link only
after the Zenodo record is published.

## Build the PDF

Use Python 3.11 or later in a separate environment. From this directory:

~~~text
python -m pip install -r pdf-requirements.txt
python build_pdf.py
~~~

The default output is output/pdf/binary-array-rqmc.pdf. Pandoc is bundled by
pypandoc_binary; Typst uses embedded fonts, so no LaTeX installation or system-font
configuration is required. The pinned package includes Pandoc 3.9.
For dependencies installed with pip --target, pass --tools-dir PATH.
The --output PATH option selects a different PDF destination.

The builder reads the manuscript and existing figure files; it does not run
simulation, calibration, validation, or timing experiments. Its layout
transformation preserves every mathematical node and pseudocode block.
pdf-style.typ controls layout. Intermediate Typst source goes under tmp/pdfs/.
After any layout or text change, render and visually inspect all affected pages.

## Rebuild the deposit package

~~~text
python build_publication.py
~~~

This rebuilds the PDF and creates a reproducibility-capsule.zip, a release
manifest, and checksums from an explicit source-file list. It accepts
--tools-dir PATH for PDF dependencies. From the development tree it writes to
publications/binary-array-rqmc/zenodo/; from an extracted capsule it writes to
output/release/. It never publishes, calls a remote API, runs new experiments,
or overwrites the earlier review ZIP.

For a separate-directory check, unzip the capsule, install the stated scientific
dependencies, and run the following from repro/:

~~~text
python -m unittest discover -v
python render_results.py
~~~

The renderer writes TABLES.md and the two PNG/SVG figure pairs under ../figures/.
Its inputs are the bundled saved data. The optional run_smoke.py prints new
engineering timings, not new manuscript results.

## Licenses

The manuscript, figures, documentation, and bundled result data are CC BY 4.0.
Code remains MPL-2.0. See [LICENSES.txt](LICENSES.txt) for exact scope and
repro/LICENSE for the full MPL text. AI assistance and the scope of within-project
verification are disclosed in the manuscript.
