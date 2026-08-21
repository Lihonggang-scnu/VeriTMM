# Changelog

## 0.6.0-upgrades-batch1 — Certificate diagnostics, sampling robustness, physics invariants, stress suite

### Certificate & Acceptance（`tmm_engine/acceptance.py`）
- Added `tightest_margin` field to every physics-acceptance certificate.
  Reports which acceptance check (energy_conservation or cross_solver_agreement)
  is closest to its failure threshold, with `observed_value`, `acceptance_limit`,
  `distance_to_limit`, and `normalized_margin`.  Field is `null` when no
  continuous check values are available.
- Added `high_precision_referee` field (informational only, never changes
  `accepted`).  Triggered when cross-solver check fails or
  `tightest_margin.normalized_margin < 0.20`.  Uses mpmath characteristic-matrix
  TMM at 113-bit (quad) precision to identify which float64 solver is closer to
  the high-precision referee reference.  Degrades gracefully to
  `{"status": "unavailable"}` when mpmath is not installed.

### Sampling robustness（`tmm_engine/research/sampling.py`）
- Sobol sampler now requires an explicit `fallback_policy` option when dimension
  exceeds the 16-D limit.  Default is `"fail"` (fail-closed, preserves declared
  semantics).  Set `options={"fallback_policy": "lhs"}` to allow automatic
  downgrade to Latin Hypercube; the candidate metadata records
  `effective_strategy`, `declared_strategy`, and `fallback_reason` so the
  strategy change is never silent.

### Physics Metamorphic Suite（`tests/test_physics_metamorphic.py`, 16 tests）
- Structural physics invariants verified without golden numbers:
  split-layer identity, zero-thickness insert, lossless R+T=1 (normal and
  oblique), passivity bounds, normal-incidence s/p equivalence, wavelength-grid
  refinement consistency, single-interface Fresnel analytic check, reciprocity.

### TMM StressBench（`tests/test_tmm_stress.py`, 17 tests）
- Numerical-stability stress tests for extreme configurations: 40-layer DBR,
  high-RI film (n=4.0), metallic film (n=0.5+3.0i), near-grazing 85° incidence
  (s and p), ultra-thin 0.5 nm film, Brewster-angle p-pol (R<0.01).
  All lossless cases verified with energy_tolerance=1e-6.

### AgentBench oblique cases（3 new cases in `benchmarks/cases/simulation/`）
- `sim_oblique_sp_equivalence`: 0° dual-polarisation, asserts certificate
  accepted and `tightest_margin` present.
- `sim_oblique_dbr_multiangle`: 4-period DBR at 0°/30°/45°, asserts certificate
  accepted at all angles.
- `sim_oblique_energy_conservation`: 45° lossless film, asserts
  `energy_conservation_max_abs_error ≤ 1e-6`.

### Optional dependency
- Added `[project.optional-dependencies] high_precision = ["mpmath>=1.3"]`.
  Install with `pip install veritmm[high_precision]`.

### Test coverage
- `tests/test_certificate_tightest_margin.py` — 7 unit tests for
  `_compute_tightest_margin`.
- `tests/test_sobol_fallback.py` — 6 tests for explicit fallback_policy
  semantics.
- `tests/test_high_precision_referee.py` — 9 tests including R+T=1 to atol=1e-12
  and float64 agreement to atol=1e-9.
- Full suite: **388 passed / 0 failed**.  AgentBench: **85/85**.

### Not added
- No new physics solvers, RCWA/FDTD/FEM, ML/RL algorithms, third-party
  optimizers, MCP, or Gymnasium dependencies.  Physics validation unchanged;
  `physics_accepted=False` cannot be bypassed.

## 0.6.0 — Research interface

- Added strict, deterministic design-space, candidate, objective, constraint,
  evaluator, batch, sampling, dataset, optimizer-adapter, optional Torch, and
  dependency-free fixed-layer environment contracts.
- Routed every research evaluation through managed simulation, existing
  preflight/capability gates, deterministic TMM, and independent certificate
  validation; scores and rewards never imply physics validity.
- Added random, bounded-grid, Latin-hypercube, and 16-dimensional core Sobol
  sampling, resumable artifact-backed datasets, bounded batch/adapter results,
  capability discovery, examples, and end-to-end acceptance tests.
- Deliberately did not add new physics, solvers, concrete third-party
  optimizers, ML/RL algorithms, PINNs, diffusion models, MCP, Gymnasium, or
  SciPy requirements.

## 0.5.2 — Context-efficient response infrastructure

- Added one protocol-level `compact`/`standard`/`full` response projection for
  CLI and Python execution, study, optimization, and benchmark responses.
- Kept `RUN_RESULT.json` as the unified first-read entry while externalizing
  spectra, wavelength grids, channel arrays, samples, optimizer histories,
  sweep children, benchmark cases/trajectories, and detailed provenance from
  every response profile; persisted run details remain in hashed artifacts.
- Added typed, actionable compact failures, deterministic context-budget
  guards, bounded artifact-reference projections, and the additive
  `veritmm-response-v1` metadata strategy with truthful artifact/profile
  availability fields, without changing task or numerical schemas.
- Added adversarial scaling and artifact-integrity regression tests, plus
  `--detail compact|standard|full` CLI documentation in both languages.

## 0.5.1 — Local hardening candidate

- Made `ExperimentStore` append-only by `run_id`, added an explicit lifecycle-only
  status update API, and rejected non-empty canonical/cache destinations instead of
  merging or replacing provenance artifacts.
- Versioned tolerance output as `veritmm-tolerance-result-v2`: `conditional_yield`
  uses completed samples, `overall_success_fraction` uses requested samples, and
  computational failures carry a four-class taxonomy. All-failed studies now report
  null conditional statistics and a typed `insufficient_valid_samples` failure.
- Prevented survivor bias in final robust selection. An incomplete or failed final
  ensemble has a null robust objective and cannot become `best_robust`, although a
  valid nominal design may remain `best_nominal`.
- Unified robust training and final evaluation around a declared normal distribution,
  `truncate` boundary policy, shared minimum physical thickness, deterministic
  training seed, and a distinct deterministic proof seed.
- Rejected non-finite, non-passive, or energy-invalid perturbation outputs before
  statistical accounting; versioned compact result summaries as v2; rewrote every
  run identity on cache replay, including sweep children and their ledger lineage;
  and made the legacy `most_robust` alias obey the formal zero-failure robust gate.
- Made formal robustness failure fail closed: nominal candidates remain available,
  but all robust winner roles stay null instead of falling back to a heuristic score.
- Split CI into core Python, real PyTorch science, lint, AgentBench, and isolated-wheel
  build jobs.

## 0.5.0 — 2026-08-20

- Added VeriTMM-AgentBench with 80+ deterministic valid, invalid, advanced,
  and out-of-scope cases; offline scoring includes failure-code accuracy,
  artifact completeness, certificate success, reproducibility, and explicit
  unsupported false-acceptance.
- Added a framework-neutral Agent A/B trajectory contract and scorer. Missing
  token/timing data remain null, and benchmark evidence is never a physics
  certificate.
- Added machine-oriented bootstrap documents for unfamiliar coding agents.
- Shipped 82 deterministic AgentBench cases, including 18 explicitly
  out-of-domain cases with zero false acceptance in the release gate, plus a
  blind-agent trajectory and independently audited result evidence.
- Verified the wheel in an isolated Python 3.11 environment: machine JSON,
  preflight, all packaged benchmark cases, and the packaged blind trajectory
  passed without importing the source checkout.
- Deliberately deferred the optional MCP transport: the core Python and CLI
  protocol remains complete without an MCP or model SDK dependency.

## 0.4.0 — Local development milestone

- Added thickness sensitivity with PyTorch autodiff and an independent NumPy
  central-finite-difference audit.
- Added seeded normal/uniform thickness uncertainty, optional correlated bias,
  yield accounting, Wilson score intervals, and explicit failed samples.
- Added stochastic robust objectives and independent final Monte Carlo
  evaluation; nominal, robust, and quantized candidate roles remain distinct.
- Kept robustness reports separate from nominal physics certificates.

## 0.3.0 — Local development milestone

- Added SQLite/filesystem `ExperimentStore`, invocation IDs, content-aware
  cache provenance, parent/child lineage, history, inspect, and compare.
- Added finite allow-listed one-dimensional and Cartesian sweeps with stable
  ordering, child artifacts, partial-failure reporting, checkpoints, and resume.
- Added deterministic scalar study metrics and experiment metadata that cannot
  affect physics-certificate identity.

## 0.2.0 — Local development milestone

- Added the deterministic, AI-facing protocol around the existing TMM engine:
  `veritmm describe --json`, simulation and optimization schema export,
  `preflight task.json --json`, and `run task.json --output-dir ... --json`.
- Added capability discovery, full-declared-grid material preflight, typed failures with
  action-safety metadata, and explicit backend routing.
- Added `RUN_RESULT.json` as the first-read run envelope and compact
  `RESULT_SUMMARY.json` so callers do not need to ingest a full spectrum before
  choosing the next action.
- Added result-side requested-output verification, full-grid material dry-run,
  independently certified optimization portfolios, and recursive candidate
  artifact hashing.
- Unified the legacy task script and installed CLI behind one execution path.
- Documented the governing separation: AI proposes, TMM computes, and the
  verifier certifies. Material extrapolation is never enabled automatically.
- Marked sweep, tolerance, sensitivity, `ExperimentStore`, external-solver
  execution, and an LLM kernel as outside the formal v0.2 protocol at that
  milestone; v0.3/v0.4 subsequently added the first four capabilities without
  adding an external solver or LLM kernel.

## 0.1.0 — 2026-08-16

- First standalone open-source release.
- Stable S-matrix, characteristic-matrix diagnostic path, and Byrnes reference backend.
- Coherent and mixed coherent/incoherent multilayer calculations.
- Material registry with explicit dataset identity, range checks, and bundled
  refractiveindex.info mirror.
- Differentiable multiband thickness optimization with independent validation.
- Capability assessment, spectral-convergence audit, and machine-readable
  physics acceptance certificates.
- Reproducible JSON task contracts, examples, and literature-trend validation.
