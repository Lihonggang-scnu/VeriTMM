# Thickness uncertainty and yield (v0.5.1)

Tolerance analysis evaluates a nominally certified planar TMM design under
explicit manufacturing-thickness uncertainty. It is a statistical study, not a
replacement for the nominal physics certificate and not a claim of certainty.

## Contract

```json
{
  "schema_version": "tolerance-task-v1",
  "mode": "tolerance",
  "tolerance": {
    "simulation": {"stack": {}, "spectrum": {}},
    "uncertainties": [
      {"layer_index": 0, "distribution": "normal", "sigma_nm": 2.0},
      {"layer_index": 1, "distribution": "uniform", "half_width_nm": 1.5}
    ],
    "metric": {"name": "mean_T", "observable": "T", "aggregation": "mean"},
    "target": {
      "metric": {"name": "mean_T", "observable": "T", "aggregation": "mean"},
      "constraint": "at_least",
      "value": 0.97
    },
    "sample_count": 200,
    "seed": 42,
    "boundary_policy": "truncate",
    "min_thickness_physical_nm": 0.1
  }
}
```

Each layer index appears at most once. Normal uncertainty requires
`sigma_nm`; uniform uncertainty requires `half_width_nm`; the two parameters
are mutually exclusive. An optional global correlated deposition bias is drawn
from a normal distribution and added to each layer.

Sampling uses NumPy’s seeded generator. The seed, distribution parameters,
drawn thicknesses, and per-sample status are retained so a study can be
replayed. Layers without an uncertainty declaration remain fixed. In v0.5.1,
the public `truncate` policy maps every raw sample to
`max(min_thickness_physical_nm, raw_thickness_nm)` before evaluation. Raw and
bounded values are both retained. A sample that still cannot be evaluated is
recorded as `invalid_perturbed_design`, `numerical_failure`, `material_failure`,
or `unexpected_runtime_failure`; it is not silently treated as a target miss.
Non-finite observables, failed passivity bounds, and failed energy audits are
computational failures even if the solver returned an object.

## Yield and statistics

`veritmm-tolerance-result-v2` separates two questions:

```text
conditional_yield = target_pass_count / completed_sample_count
overall_success_fraction = target_pass_count / requested_sample_count
```

The first estimates scientific target attainment conditional on successful
computation. The second is an end-to-end operational fraction that also
reflects computational failures. The v2 compatibility aliases `yield` and
`target_pass_probability` equal `conditional_yield`; `yield_ci95` equals
`conditional_yield_ci95`. Consumers of v1 that assumed a requested-sample
denominator must migrate to `overall_success_fraction`. The result reports:

- mean and standard deviation;
- p01, p05, p50, p95, and p99;
- worst-case metric under the declared constraint;
- conditional yield, overall success fraction, and target pass count;
- sample, completed, and failed counts; and
- the computational failure taxonomy; and
- the random seed and full sample records.

`conditional_yield_ci95` is a two-sided Wilson score interval whose denominator
is `completed_sample_count`. If every requested sample fails computationally,
conditional yield, its interval, and distribution statistics are null and the
run records `insufficient_valid_samples`; VeriTMM does not fabricate zero yield
from an empty valid ensemble.

## Separate artifacts

Tolerance emits both:

```text
PHYSICS_ACCEPTANCE_CERTIFICATE.json
TOLERANCE_RESULT.json
ROBUSTNESS_REPORT.json
```

The certificate describes the nominal TMM result. The tolerance result and
robustness report describe statistical behavior around that result. A nominal
physics-valid design may have zero yield or a wide confidence interval, and a
robustness report may not rewrite `accepted` or the certificate ID.

Unsupported geometry, material models, excitation, time-domain requests,
material extrapolation failures, and invalid task declarations remain rejected
by the existing verifier boundary.
