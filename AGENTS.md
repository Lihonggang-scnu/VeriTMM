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
