# Thickness uncertainty and yield (v0.4)

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
    "seed": 42
  }
}
```

Each layer index appears at most once. Normal uncertainty requires
`sigma_nm`; uniform uncertainty requires `half_width_nm`; the two parameters
are mutually exclusive. An optional global correlated deposition bias is drawn
from a normal distribution and added to each layer.

Sampling uses NumPy’s seeded generator. The seed, distribution parameters,
drawn thicknesses, and per-sample status are retained so a study can be
replayed. Layers without an uncertainty declaration remain fixed. A sample
that becomes invalid or cannot be evaluated is recorded as `failed` with a
typed failure; it is not counted as a success.

## Yield and statistics

Yield is the explicit target pass count divided by the requested sample count.
Failed samples remain in that denominator. The result reports:

- mean and standard deviation;
- p01, p05, p50, p95, and p99;
- worst-case metric under the declared constraint;
- target pass probability and `yield`;
- sample, completed, and failed counts; and
- the random seed and full sample records.

`yield_ci95` is a two-sided Wilson score interval with the standard 95% normal
quantile. It is labeled `wilson_score_interval`; it is not an unlabeled
confidence score.

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

