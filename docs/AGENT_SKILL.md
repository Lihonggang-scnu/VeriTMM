# VeriTMM v0.6 AI-Facing TMM Skill

Use this skill only after a scientific request has been converted into a
standard simulation, optimization, sweep, sensitivity, or tolerance task.
This is a deterministic interface to the TMM engine, not a generic scientific-
research workflow. Do not infer missing units or scientific choices.

## Shortest safe call order

1. Discover the allow-list:

   ```bash
   veritmm describe --json
   ```

2. Inspect the relevant contract before generating JSON:

   ```bash
   veritmm schema simulation
   veritmm schema optimization
   veritmm schema sweep
   veritmm schema sensitivity
   veritmm schema tolerance
   ```

3. Create one complete task. Wavelengths and thicknesses are in nanometres;
   angles are in degrees. Select material datasets explicitly when identity
   matters.

4. Gate the task without running the full calculation:

   ```bash
   veritmm preflight task.json --json
   ```

   Preflight validates the contract, capability boundary, material coverage on
   the complete declared wavelength grid, backend routing, and numerical risk notices. It
   does **not** run a complete spectrum or an optimization. If it is rejected,
   stop and follow the typed failures instead of guessing a repair.

5. Execute only a ready task:

   ```bash
   veritmm run task.json --output-dir outputs/tmm_run --json
   ```

6. Read `RUN_RESULT.json` first. Then read `RESULT_SUMMARY.json` for compact
   physics and spectral features. Inspect
   `PHYSICS_ACCEPTANCE_CERTIFICATE.json`, `SIMULATION_RESULT.json`, and
   `SPECTRA.csv` only when the next scientific decision needs their detail.

The governing division is: **AI proposes, TMM computes, and the verifier
certifies**. A differentiable optimizer may propose thicknesses, but its
candidate must be independently recomputed before certification.

## Non-negotiable rules

1. Wavelength and thickness fields in the public JSON contract are nanometres;
   angles are degrees.
2. A finite substrate is a finite `LayerSpec` followed by an exit medium. Never
   set a flag and silently treat it as semi-infinite.
3. Select material datasets by provider and `dataset_id` when reproducibility
   matters. Optical-constant extrapolation is not enabled automatically. A
   range failure requires a covered dataset or an explicit scientific decision;
   do not invent an extrapolation patch.
4. `A = 1 - R - T` means absorption in finite layers, where `T` is power entering
   the semi-infinite exit medium. `E_system = 1 - R` is a different quantity and
   is valid only when the absorbing exit medium belongs to the emitting system.
5. Use the stable S-matrix backend by default. Use the characteristic matrix
   only for diagnosis and the Byrnes backend for mixed coherent/incoherent
   stacks, field profiles, layer absorption, or ellipsometry.
6. Accept an optimization only when `INDEPENDENT_VALIDATION.json` has status
   `passed` and the final physics certificate has `accepted: true`. An
   optimizer cannot certify itself.
7. Never clip raw spectra to hide negative absorption, non-finite values, or
   energy-balance violations. Treat them as diagnostics.
8. Request coherent amplitudes from `smatrix` or `characteristic`, not the
   Byrnes route. Phase dispersion requires at least three wavelength samples;
   nine or more are recommended for useful numerical derivatives.
9. Do not treat a successful optimizer status as a unique scientific answer.
   Read `DESIGN_PORTFOLIO.json` and the independently certified candidate
   artifacts before selecting a design role.
10. A physics certificate establishes numerical/physical acceptance of the
    nominal task. It does not establish high yield, low sensitivity, or robust
    optimality. Read the corresponding study report separately.
11. Use `--resume` only for a sweep. A cache hit must carry a new invocation
    run ID and point back to its source run; never present it as a new solve.

## Response profiles and progressive disclosure

CLI and machine-facing Python responses default to `compact` so the first
observation contains only decision-critical status, certificate identity,
bounded metrics, artifact references, and typed next actions:

```bash
veritmm run task.json --output-dir outputs/tmm_run --json --detail compact
veritmm run task.json --output-dir outputs/tmm_run --json --detail standard
veritmm run task.json --output-dir outputs/tmm_run --json --detail full
```

`compact` is also the default when `detail` is omitted from execution response
APIs. It targets 16 KiB and has a fixed 32 KiB hard limit. `standard` exposes
bounded diagnostic collections and `full` preserves richer scalar/mapping
metadata and context. Full is the full view of retained, bounded metadata,
not raw data: every profile externalizes spectra,
wavelength grids, channel arrays, sample lists, optimization histories, sweep
children, benchmark cases/trajectories, and full provenance. All three profiles
keep the same unified `RUN_RESULT.json` entry and the same task/certificate
identities.

The profile is additive protocol metadata under `summary.response`, versioned
as `veritmm-response-v1`; the existing `veritmm-run-result-v1` envelope and
task schemas remain valid for older consumers. Each run persists a bounded,
unprojected `RESPONSE_CONTEXT.json`, versioned as
`veritmm-response-context-v2`. It declares fixed retention limits and explicit
omission/truncation accounting. `inspect --detail standard|full` reads that
source and never re-projects an already compact `RUN_RESULT.json`; its entire
v2 response has one outer profile. Legacy runs without a validated context
return typed `response_detail_unavailable` for richer profiles. Detailed
data is never thrown away: read the relative, SHA-256-checked references in
`RUN_RESULT.json` to open `SPECTRA.csv`, study result documents, candidate
artifacts, or the full preflight/certificate context. Compact output
deliberately contains no spectra, samples, history, sweep children, detailed
provenance, or trajectory arrays. Counts and omission metadata are navigation
hints, not replacements for the artifact. `artifact_backed` is false when the
response carries no reachable artifact reference; `detail_available_via_profile`
only describes richer non-array context available from another profile.

## Typed failures and action safety

Every typed failure may include `recoverable`, `requires_user_choice`, and
`actions`. Each action declares one of:

- `safe`: a mechanically safe execution or diagnostic step;
- `requires_scientific_judgment`: the agent or a domain expert must decide what
  the scientific change means;
- `requires_user_input`: a user must choose a material, dependency, or other
  external change.

Recoverable does not mean safe to patch automatically. Never apply a
scientific material, wavelength, geometry, solver-family, or target choice just
because it appears in a failure message. In particular, a material-range
failure never turns extrapolation on by itself.

In compact responses, failure `code`, `message`, severity, choice requirement,
and action-safety metadata remain inline; verbose failure context is
artifact-backed. Follow the action safety field and request `standard` or
`full` only when the detailed artifact context is needed.

## Material lookup when needed

Search the bundled refractiveindex.info mirror and export an exact dataset:

```powershell
python scripts/search_optical_material.py TiO2 --provider rii `
  --start-nm 500 --stop-nm 800 --dataset-id 418 `
  --export-csv outputs/tio2.csv
```

Use the selected provider and `dataset_id` in the task when reproducing a
result. The engine does not silently substitute another material.

## Supported TMM task families

- coherent thin-film stacks;
- thick/incoherent substrates and mixed-coherence stacks;
- periodic 1D photonic crystals, defect cavities, and chirped stacks;
- wavelength-, angle-, and polarization-resolved R/T/A;
- system emissivity with explicit semantics;
- position-resolved fields and absorption;
- finite-layer absorption and ellipsometry;
- Bloch forbidden-band analysis;
- multi-band, multi-angle differentiable thickness optimization with equality,
  lower-bound, or upper-bound targets, mean or worst-case aggregation, fixed
  layers, fabrication bounds, multistart, and thickness quantization.
- finite allow-listed parameter sweeps with checkpoint/resume;
- thickness sensitivity with independent finite-difference audit;
- seeded thickness tolerance, yield, and Wilson confidence intervals;
- robust stochastic thickness optimization with independent final Monte Carlo.

## Study and benchmark commands

Use `history`, `inspect`, `lineage`, and `compare` for persisted experiments.
Use `benchmark --offline --json` to evaluate the protocol without an LLM or
network. Benchmark evidence never changes a physics gate. An external agent
runner may emit the documented A/B trajectory format; unavailable token or
timing fields must remain null rather than being estimated.

## Research-interface call order

Use `DesignSpace` only as a deterministic candidate-to-`SimulationTask`
adapter. Submit candidates through `ResearchEvaluator` or its public batch
path; use `DatasetFactory` for sampled datasets. Never call a workbench,
solver, or certifier from an external algorithm.

Read `EvaluationRecord.physics_accepted`, `certificate_id`, `run_id`, and
`task_sha256` together. A score, feasible flag, optimizer winner, Torch target,
or environment reward cannot replace those fields. Dataset and batch responses
are compact and reference detailed run artifacts. Sampling is deterministic by
design-space identity, plan, seed, and sample index; the core Sobol sampler
supports at most 16 dimensions. Fixed-layer `add_layer` and `remove_layer`
actions are typed unsupported in v0.6.

## Physics boundary

This is an isotropic planar 1D solver. It does not claim to model lateral
patterning, diffraction gratings, metasurface unit cells, diffuse scattering,
anisotropic tensor layers, nonlinear optics, thermal transport, or fabrication
chemistry. Route those tasks to a suitable RCWA/FDTD/FEM/thermal tool outside
the VeriTMM v0.6 protocol; VeriTMM will not perform that handoff or certify the
external result.

VeriTMM does not execute external solver families and contains no LLM kernel.
The optional MCP adapter is deferred; use the complete CLI or Python protocol.
