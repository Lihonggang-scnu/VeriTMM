# Validation philosophy

VeriTMM separates **target quality** from **physical admissibility**. A design
can miss an ambitious target and still be a valid simulation; it may not pass
the physics gate by merely achieving a high optimization score. The v0.5
protocol adds deterministic transport, experiment identity, scientific-study
semantics, and failure contracts around the TMM
calculation; it does not replace the TMM checks.

The governing roles are: **AI proposes, TMM computes, and the verifier
certifies**. An optimizer is never its own certificate authority.

## Protocol-level validation

The public command surface is intentionally small and machine-readable:

```text
veritmm describe --json
veritmm schema simulation
veritmm schema optimization
veritmm schema sweep
veritmm schema sensitivity
veritmm schema tolerance
veritmm preflight task.json --json
veritmm run task.json --output-dir outputs/run --json
veritmm benchmark --offline --json
```

`preflight` checks the task contract, capability boundary, material coverage on
the complete declared wavelength grid, backend routing, and numerical risk notices. It does
**not** run a complete spectrum or optimization. `run` repeats the preflight
before computation and always produces `RUN_RESULT.json` for a normal run or a
preflight rejection. Typed failures carry action safety metadata; a recoverable
failure is not permission to apply a scientific change automatically.

## Acceptance layers

1. **Contract** — the JSON task must match the simulation or optimization
   contract and the runtime task model.
2. **Capability boundary** — the task must be planar, one-dimensional,
   isotropic, linear, and compatible with a plane-wave frequency-domain model.
3. **Material validity** — every optical-constant dataset must cover the
   requested wavelength range. Extrapolation is not enabled automatically; a
   range exception requires an explicit scientific decision.
4. **Preflight routing** — the requested outputs and coherence model must map to
   a supported backend before any full spectrum is executed.
5. **Raw numerical checks** — non-finite spectra, negative passive absorption,
   and energy-balance violations are surfaced rather than clipped away.
6. **Spectral convergence** — the result is repeated on refined wavelength grids.
7. **Independent comparison** — coherent results are compared with the Byrnes
   implementation when the task supports it. Optimization proposals are
   independently recomputed before acceptance.
8. **Certificate and envelope** — checks, tolerances, material identities,
   hashes, and limitations are serialized in
   `PHYSICS_ACCEPTANCE_CERTIFICATE.json`; `RUN_RESULT.json` exposes the status,
   failure actions, and artifact references.

## First-read artifacts

Read `RUN_RESULT.json` first. It is the run envelope, not a replacement for the
scientific outputs. Read `RESULT_SUMMARY.json` next for compact physics and
spectral features so an agent does not have to ingest the full
`SIMULATION_RESULT.json` or `SPECTRA.csv` before choosing its next action. Read
the certificate and full spectra when detailed scientific inspection is needed.

## Current deterministic coverage

The regression suite contains tests for:

- Fresnel and lossless-stack invariants;
- S-matrix/characteristic/reference-solver agreement;
- random passive multilayer regression;
- finite and mixed-coherence stacks;
- material ambiguity, dataset identity, interpolation, and range rejection;
- differentiable forward agreement and gradient optimization when PyTorch is installed;
- protocol capability manifests, JSON Schema exports, preflight without solver execution,
  typed failure actions, and single-object CLI output;
- run-result envelopes, artifact hashes, compact summaries, and preflight rejection;
- task serialization and command-line execution;
- experiment identity, cache provenance, lineage, compare, sweep and resume;
- autodiff sensitivity independently audited by NumPy finite differences;
- seeded tolerance sampling with separate conditional-yield and operational-success
  denominators, typed computational failures, and Wilson intervals over completed samples;
- robust candidate evaluation separated from nominal physics certificates, including a
  zero-failure completeness gate that prevents survivor-biased robust selection;
- offline AgentBench case contracts, reproducibility and unsupported false-acceptance;
- a published DBR/defect-cavity trend reproduction based on PMC9147317.

The literature case checks stop-band overlap, cavity-dip position, and field
enhancement trends. It is a trend-level reproduction, not a claim of exact
fabrication-level replication.

## Deliberate protocol boundaries

Sweep, sensitivity, tolerance, and robust optimization are formal v0.5 study
operations, but their reports remain distinct from physical validity. The
protocol does not execute external solvers or contain an LLM kernel. AgentBench
is evaluation evidence rather than a physics certificate, and the optional MCP
transport is deferred. These boundaries do not weaken the TMM checks; they
define what each artifact is allowed to claim.
