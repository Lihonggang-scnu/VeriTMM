# Agent bootstrap for VeriTMM

VeriTMM is a verifier-first execution environment for passive, isotropic,
planar multilayer optics. It is not a general Maxwell solver. Do not infer the
public task format from numerical source code.

## Required first actions

From the repository root, discover the installed contract before constructing
a task:

```bash
veritmm describe --json
veritmm schema simulation
veritmm schema optimization
veritmm schema sweep
veritmm schema sensitivity
veritmm schema tolerance
```

If the console script is unavailable, replace `veritmm` with
`python -m tmm_engine.cli`.

For every proposed task:

1. Write one JSON task using the exported schema and nanometre/degree units.
2. Run `veritmm preflight TASK.json --json`.
3. If rejected, read the typed failure code and each action's `safety` field.
   Never apply a material, target, geometry, extrapolation, or solver-family
   change silently.
4. Only after preflight is ready, run
   `veritmm run TASK.json --output-dir RUN_DIR --json`.
5. Treat `RUN_RESULT.json` as the artifact index and
   `RESULT_SUMMARY.json` as the compact reading surface.
6. A successful optimizer does not certify itself. Require an accepted
   `PHYSICS_ACCEPTANCE_CERTIFICATE.json` and independent validation.
7. Robustness and nominal physics validity are separate claims.

## Response profiles and context budget

Machine-facing CLI and Python execution responses use the `compact` profile by
default. Select a richer projection explicitly with
`--detail standard` or `--detail full` (Python entry points accept the same
`detail=` value). The profile applies to the response envelope only; the full
`SIMULATION_RESULT.json`, `OPTIMIZATION_RESULT.json`, `SWEEP_RESULT.json`,
`SENSITIVITY_RESULT.json`, `TOLERANCE_RESULT.json`, benchmark result, and
certificate artifacts are still written.

`RUN_RESULT.json` remains the single unified first-read entry. Its artifact
references are relative, hashed, and size-checked. Compact responses target
16 KiB and have a fixed 32 KiB hard limit. They keep decision-critical status,
certificate identity, counts, typed failure actions, and bounded summaries.
Standard and full add bounded/richer scalar and mapping context. Full is the
full view of retained, bounded metadata, not raw data: spectra, wavelength
grids, channel arrays, samples, optimizer
histories, sweep children, benchmark cases/trajectories, and detailed
provenance are externalized for every profile. `artifact_backed` is true only
when reachable references are present; `detail_available_via_profile`
describes richer non-array context available from another profile. The nested
`summary.response` object uses `veritmm-response-v1`; the bounded, unprojected
`RESPONSE_CONTEXT.json` source uses `veritmm-response-context-v2` and records
its retention limits plus omitted/truncated paths so
`inspect --detail standard|full` can reconstruct richer profiles without
re-projecting compact `RUN_RESULT.json`. The complete inspect document has one
outer response profile and compact guard. A legacy run without a validated
response context returns typed `response_detail_unavailable` for a richer
request. The existing
`veritmm-run-result-v1` envelope and task hashes remain backward compatible.

For diagnostics or a legacy-style detailed read, use the artifact references
or request `--detail standard|full`. Never infer a missing large array from a
compact count; open the referenced artifact instead.

## Hard boundaries

Reject rather than approximate lateral gratings, metasurface unit cells,
arbitrary 2D/3D geometry, anisotropic/tensor or nonlinear media, finite beams,
dipoles, mode sources, and time-domain requests. VeriTMM may suggest another
solver family but does not execute or certify that handoff.

Never enable material extrapolation, substitute a material dataset, remove a
requested output, or weaken a physics gate merely to obtain a successful run.

## Stateful experiments and studies

Use `--store-dir`, `--experiment-id`, `--parent-run-id`, `--hypothesis`, and
`--change-reason` to preserve experiment lineage. Use `history`, `inspect`,
`lineage`, and `compare` rather than comparing files by eye. Cache replay must
retain a new invocation ID and its source run ID. `--resume` is valid only for
sweeps.

## Repository checks

Before submitting code changes, run:

```bash
python -m pytest -q
python -m ruff check .
veritmm benchmark --offline --json
```

The release benchmark must contain at least 80 cases and must report exactly
zero false acceptance for unsupported physics. Benchmark and agent-trajectory
evidence never changes a physics certificate.

More detail: `docs/AGENT_SKILL.md`, `docs/AGENTBENCH.md`, and
`docs/VALIDATION.md`.
