# VeriTMM v0.4 implementation report

Date: 2026-08-20

## Scope

This handoff adds only the v0.4 verification and documentation files assigned
to this worker. No core file and no v0.3 file was edited while the main agent
integrated v0.4.

The owned tests cover:

- sensitivity and tolerance JSON contracts, round trips, JSON Schema, and
  fail-closed invalid fields;
- a simple uniform-medium analytical zero-reflection/zero-gradient case;
- non-zero PyTorch autodiff versus independent central finite difference;
- near-zero-gradient absolute-tolerance handling and null relative error;
- exclusion of fixed layers from sensitivity rows and ranking;
- reproducible seeded normal and uniform draws, different-seed divergence, and
  fixed-layer preservation;
- Wilson score intervals and yield accounting;
- failed tolerance samples retained with typed failures and included in the
  requested-trial denominator;
- strict separation of nominal physics certificates from tolerance robustness
  reports; and
- a deterministic portfolio in which the nominal-best and robust-best roles
  differ, while ineligible candidates remain excluded.

## Deterministic evidence

Focused command executed with Python 3.11 and CPU PyTorch:

```text
py -3.11 -m pytest -q \
  tests/test_sensitivity_schema.py \
  tests/test_sensitivity_autodiff.py \
  tests/test_sensitivity_finite_difference.py \
  tests/test_tolerance_schema.py \
  tests/test_uncertainty_sampling.py \
  tests/test_yield_statistics.py \
  tests/test_robustness_report.py \
  tests/test_robust_optimization.py
```

Final focused result: **25 passed in 2.86s**.

The tests use only constant-index stacks, local NumPy/PyTorch, and deterministic
seeds. No network, external solver, LLM, RCWA, FDTD, FEM, or proprietary API
was used.

## Scientific integrity checks

The suite locks these v0.4 boundaries:

1. autodiff is independently audited by central finite difference;
2. near-zero derivatives use an absolute tolerance rather than a misleading
   relative ratio;
3. fixed layers are not silently ranked as sensitivity parameters;
4. normal and uniform sampling preserve declared seed/distribution provenance;
5. Wilson intervals are explicitly labeled and mathematically reproducible;
6. failed samples are not counted as successes;
7. a physics certificate remains separate from yield and robustness artifacts;
8. only independently validated, physics-admitted candidates can receive
   formal robust roles; and
9. nominal and robust role selection can produce different candidates.

## Exact owned artifacts

- `tests/test_sensitivity_schema.py`
- `tests/test_sensitivity_autodiff.py`
- `tests/test_sensitivity_finite_difference.py`
- `tests/test_tolerance_schema.py`
- `tests/test_uncertainty_sampling.py`
- `tests/test_yield_statistics.py`
- `tests/test_robustness_report.py`
- `tests/test_robust_optimization.py`
- `docs/SENSITIVITY.md`
- `docs/UNCERTAINTY.md`
- `docs/ROBUST_OPTIMIZATION.md`
- `V0_4_IMPLEMENTATION_REPORT.md`

## Integrated release-gate status

The historical focused evidence above is now complemented by the integrated
v0.5 release-candidate gates: the full repository suite and Ruff pass, all
seven bundled task examples execute successfully, the wheel builds and installs
in an isolated Python 3.11 environment, and the packaged offline benchmark
passes. Sensitivity, tolerance, and robust optimization are represented in the
release catalogue and remain separate from nominal physics certification.
