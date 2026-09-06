# Capability boundaries and material governance

## Supported physics (fail-closed everywhere else)

Supported: passive, isotropic, planar 1-D multilayers — coherent and mixed
coherence stacks, thin films, DBRs, 1-D photonic crystals, defect cavities,
chirped stacks, lossy absorbers, finite substrates; multi-wavelength,
multi-angle, s/p and unpolarized illumination; R/T/A, amplitudes, fields,
layer absorption, ellipsometry, system emissivity, phase and dispersion.

Rejected with typed failures (never approximated):

- lateral gratings and metasurface unit cells (`unsupported_geometry`,
  suggested family: RCWA)
- anisotropic/tensor material models (`unsupported_material_model`)
- magneto-optic and nonlinear media (`unsupported_material_model`)
- finite beams, dipoles, mode sources (`unsupported_excitation`)
- time-domain requests (`time_domain_required`)

When a request is rejected, do not attempt a workaround, a different phrasing,
or a parameter patch. Either reformulate the task inside the supported domain
or hand the problem to an appropriate solver family outside VeriTMM.

## Material governance

- Optical constants come from governed datasets (bundled
  refractiveindex.info snapshot plus local registry CSVs). The registry
  records provider, dataset id, wavelength coverage, and provenance.
- Extrapolation beyond a dataset's measured range is **never enabled
  automatically**. A coverage failure requires a covered dataset or an
  explicit scientific decision recorded by the operator.
- Material substitutions are never silent; the capability gate rejects
  unknown or ambiguous material names.

## Execution boundaries

- `run` on the MCP surface is simulate-mode only, with default acceptance
  settings. Optimization, sweeps, sensitivity, tolerance, and robust-design
  studies are CLI/operator work outside this skill.
- Operator and debug parameters (certificate skipping, physics subprocess
  selection, convergence tolerances, device selection, output/store
  locations) are unavailable to agents by construction.
- The experiment store is append-only: runs are never overwritten, and cache
  replays receive fresh run identities linked to their sources.
