# Validation philosophy

VeriTMM separates **target quality** from **physical admissibility**.
A design can miss an ambitious target and still be a valid simulation; it may
not pass the physics gate by merely achieving a high optimization score.

## Acceptance layers

1. **Capability boundary** — the task must be planar, one-dimensional,
   isotropic, linear, and compatible with a plane-wave frequency-domain model.
2. **Material validity** — every optical-constant dataset must cover the
   requested wavelength range unless extrapolation is explicitly authorized.
3. **Raw numerical checks** — non-finite spectra, negative passive absorption,
   and energy-balance violations are surfaced rather than clipped away.
4. **Spectral convergence** — the result is repeated on refined wavelength grids.
5. **Independent comparison** — coherent results are compared with the Byrnes
   implementation when the task supports it.
6. **Certificate** — the checks, tolerances, material identities, hashes, and
   limitations are serialized in `PHYSICS_ACCEPTANCE_CERTIFICATE.json`.

## Current regression suite

The standalone suite contains deterministic tests for:

- Fresnel and lossless-stack invariants;
- S-matrix/characteristic/reference-solver agreement;
- random passive multilayer regression;
- finite and mixed-coherence stacks;
- material ambiguity, dataset identity, interpolation, and range rejection;
- differentiable forward agreement and gradient optimization when PyTorch is installed;
- task serialization and command-line execution;
- a published DBR/defect-cavity trend reproduction based on PMC9147317.

The literature case checks stop-band overlap, cavity-dip position, and field
enhancement trends. It is a trend-level reproduction, not a claim of exact
fabrication-level replication.
