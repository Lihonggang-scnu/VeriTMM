# Thickness sensitivity (v0.4)

Sensitivity is a scientific-analysis task over the existing planar, isotropic,
scalar TMM runtime. It reports how a declared scalar metric changes with
finite-layer thickness. It does not change the nominal physics acceptance
decision.

## Contract

```json
{
  "schema_version": "sensitivity-task-v1",
  "mode": "sensitivity",
  "sensitivity": {
    "simulation": {"stack": {}, "spectrum": {}},
    "metric": {
      "name": "mean_R",
      "observable": "R",
      "wavelength_min_nm": 500,
      "wavelength_max_nm": 600,
      "aggregation": "mean",
      "angle_deg": 0,
      "polarization": "unpolarized"
    },
    "parameters": "optimizable_thicknesses",
    "finite_difference_step_nm": 0.01,
    "relative_error_tolerance": 0.001,
    "absolute_error_tolerance": 0.0000001
  }
}
```

Only layers marked `optimizable` participate. Fixed layers are retained in the
nominal stack but are excluded from the derivative table and ranking. The
metric must reference a declared angle/polarization channel and overlap the
wavelength grid. Differentiable metrics are `mean`, `min`, `max`, `worst_case`,
and `value_at_wavelength` over `R`, `T`, or `A`. `E_system` and
`threshold_band_width` are not silently approximated for sensitivity.

## Independent audit

The result uses the PyTorch differentiable S-matrix backend for the derivative
and independently evaluates a NumPy central difference:

```text
[f(x + h) - f(x - h)] / (2 h)
```

Each variable layer records the step, autodiff derivative, finite-difference
derivative, absolute and relative errors, and `audit_passed`. Near-zero
gradients use the declared absolute tolerance and report `relative_error` as
`null`; this avoids an unstable relative-error ratio around zero. A normal
non-zero-gradient case must satisfy the relative tolerance before the study is
marked passed.

## Artifacts and certificate boundary

The output includes:

```text
NORMALIZED_TASK.json
PHYSICS_ACCEPTANCE_CERTIFICATE.json
SENSITIVITY_RESULT.json
RESULT_SUMMARY.json
RUN_RESULT.json
```

The nominal simulation is certified by the ordinary verifier first. The
sensitivity result is an analysis artifact and cannot sign or rewrite that
certificate. A failed finite-difference audit is not converted into a physics
failure or hidden by changing the certificate.

Sensitivity remains fail-closed for mixed-coherence stacks, unsupported physics
declarations, invalid material coverage, missing PyTorch, invalid metric
channels, and non-differentiable metric requests.

