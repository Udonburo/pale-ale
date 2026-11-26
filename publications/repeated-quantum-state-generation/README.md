# Repeated quantum state generation from finite memory

Aoi Kawasaki · Author preprint v1.0.0, 26 November 2026 · Submitted to
*Physical Review A* as a Regular Article.

[Preprint (PDF)](https://zenodo.org/records/24540440/files/repeated-quantum-state-generation-pra-preprint.pdf) ·
[Certificates and verification code (ZIP)](https://zenodo.org/records/24540440/files/repeated-quantum-state-generation-certificates-v1.0.0.zip?download=1) ·
[Version DOI](https://doi.org/10.5281/zenodo.24540440)

The device prepares one state for each request in a fixed, externally chosen
sequence. Each output must meet a prescribed marginal trace-distance bound,
and different outputs may be correlated. For a finite source family and a
finite request alphabet with unrestricted joint initialization, the least
retained memory dimension for indefinite service equals the fewest outcomes
of a single source measurement that permits request-dependent preparation
at the same error.

## Check the certificates

Extract the companion ZIP and run the following command from the extracted
directory. The checker uses only the Python standard library.

```sh
python -B checkers/check_certificates.py
```

The Python and Rust checkers independently verify the same 20 rational
certificates. In the ZIP, `ARTICLE_PROOFS.md` maps the article's theorems to
their proofs and assumptions. `REPRODUCTION.md` covers the ten finite-size
diagnostics and the Isabelle/HOL and Lean 4 proofs of selected algebraic and
spectral steps. The archive's `README.md` gives the command to regenerate
Figure 1 from the analytic bounds and four certified intervals.

The paper, figures, documentation, and data are licensed under CC BY 4.0.
The software and formal proof sources are available under MIT OR Apache-2.0.
