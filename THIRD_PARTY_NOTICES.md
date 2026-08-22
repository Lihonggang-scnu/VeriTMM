# Third-party notices and data provenance

VeriTMM is distributed under the Apache License 2.0. The transfer-matrix method
itself is established scientific knowledge; this project does not claim
invention of the underlying TMM equations.

## SpecFormer lineage

The initial NumPy and PyTorch S-matrix implementations were generalized from
the authors' SpecFormer codebase, then extended here with task contracts,
arbitrary spectral grids and stacks, governed materials, independent
recomputation, convergence audits, and acceptance certificates.

- Repository: https://github.com/Lihonggang-scnu/SpecFormer

Both projects are maintained by the same project owner; the code included in
this repository is released under this repository's Apache License 2.0.

## refractiveindex.info database

The bundled SQLite optical-constants catalog and the compact CSV tables are
derived from the refractiveindex.info database. The database is dedicated to
the public domain under CC0 1.0 Universal.

- Project: https://refractiveindex.info/
- Source repository: https://github.com/polyanskiy/refractiveindex.info-database
- License: https://creativecommons.org/publicdomain/zero/1.0/
- Recommended citation: M. N. Polyanskiy, “Refractiveindex.info database of
  optical constants,” *Scientific Data* 11, 94 (2024),
  https://doi.org/10.1038/s41597-023-02898-2.

Individual datasets retain their source-publication metadata inside the
catalog. Scientific users should cite both the database descriptor and the
original optical-constant measurement selected for a simulation.

## Byrnes `tmm`

The optional independent-reference and mixed-coherence paths use Steven J.
Byrnes' `tmm` Python package at runtime. That package is MIT-licensed and is
installed as a dependency; its source is not vendored here.

- Repository: https://github.com/sbyrnes321/tmm
- Method paper: https://arxiv.org/abs/1603.02720
