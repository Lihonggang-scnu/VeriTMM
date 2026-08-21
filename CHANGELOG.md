# Changelog

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
