# VeriTMM v0.3 implementation report

Date: 2026-08-20

## Scope

This handoff covers the v0.3 verification and documentation files owned by
this worker. Core implementation files were left untouched because they are
being changed concurrently by the main agent.

The owned regression suite covers the public v0.3 surface:

- SQLite/filesystem `ExperimentStore` records, filtering, archival, and
  reference validation;
- normalized task identity versus per-invocation run identity;
- parent/child and sweep lineage;
- deterministic task/material/solver/summary/certificate/artifact compare
  deltas;
- cache hits, new cache-hit run IDs, source provenance, execution-setting
  invalidation, and material-catalog invalidation;
- versioned sweep schema, allow-listed paths, metric shape validation, 1D and
  Cartesian expansion, stable ordering, child artifacts, partial failure, and
  resume; and
- machine-readable JSON stdout plus the legacy store-optional single-run API.

The tests also explicitly verify that research metadata is stored without
changing a physics acceptance certificate, and that unsupported physics stays
rejected even when caller metadata requests an override.

## Deterministic evidence

Focused command executed with the installed Python 3.11 environment:

```text
py -3.11 -m pytest -q \
  tests/test_experiment_store.py \
  tests/test_run_identity.py \
  tests/test_run_lineage.py \
  tests/test_run_compare.py \
  tests/test_run_cache.py \
  tests/test_sweep_schema.py \
  tests/test_sweep_execution.py \
  tests/test_sweep_resume.py
```

Final focused result: **34 passed in 5.98s**.

The default `python` command resolves to Python 3.9.7 in this workstation and
does not have `pytest` installed; the focused run therefore used `py -3.11`.
No network, cloud solver, LLM, RCWA, FDTD, FEM, or proprietary API was used.

## Physics integrity

The verification contract keeps these boundaries explicit:

1. research metadata is provenance only and is absent from the certificate
   payload and certificate identity;
2. cache identity includes normalized task, package/protocol versions,
   material catalog identity, and execution settings;
3. material/catalog changes do not reuse a prior cache result;
4. optimizer or study orchestration does not sign a physics certificate; and
5. unsupported geometry, material model, excitation, and time-domain requests
   are rejected before a TMM spectrum is executed.

## Exact owned artifacts

- `tests/test_experiment_store.py`
- `tests/test_run_identity.py`
- `tests/test_run_lineage.py`
- `tests/test_run_compare.py`
- `tests/test_run_cache.py`
- `tests/test_sweep_schema.py`
- `tests/test_sweep_execution.py`
- `tests/test_sweep_resume.py`
- `docs/EXPERIMENTS.md`
- `docs/SWEEPS.md`
- `V0_3_IMPLEMENTATION_REPORT.md`

## Integrated release-gate status

The historical focused evidence above is now complemented by the integrated
v0.5 release-candidate gates: the full repository suite and Ruff pass, all
seven bundled task examples execute successfully, the wheel builds and installs
in an isolated Python 3.11 environment, and the packaged offline benchmark
passes with no unsupported-physics false acceptance. The configured release
environment is Python 3.11; Python 3.9 remains outside the package requirement.
