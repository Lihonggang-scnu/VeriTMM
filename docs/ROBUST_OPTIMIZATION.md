# Robust optimization (v0.4)

Robust optimization evaluates independently validated optimization candidates
under explicit thickness perturbations. The optimizer proposes candidates; the
ordinary physics verifier certifies nominal candidates; the formal robustness
evaluator reports expected or worst-case behavior. These are separate roles.

## Robustness settings

```json
{
  "robustness": {
    "enabled": true,
    "objective": "mean_plus_k_sigma",
    "samples_per_step": 8,
    "final_samples": 128,
    "seed": 42,
    "thickness_sigma_nm": 2.0,
    "k_sigma": 2.0
  }
}
```

Supported formal objectives are:

- `expected_loss` — mean perturbed loss;
- `worst_case_loss` — maximum completed perturbed loss; and
- `mean_plus_k_sigma` — mean plus the declared multiple of standard deviation.

Final robustness evaluation uses an independent NumPy S-matrix sampling path.
Training-time Monte Carlo is recorded as training context and is not treated as
the final robustness proof. Failed perturbed samples are retained in the
candidate record with typed failures.

## Candidate eligibility and roles

Only candidates with both `independent_validation_status: "passed"` and a
physics status of `physically_valid` or `physically_valid_with_limits` are
eligible for formal robustness selection. The evaluator exposes interpretable
roles rather than an opaque aggregate score:

```text
best_nominal   lowest nominal objective loss
best_robust    lowest formal robustness objective
best_quantized lowest formal robustness objective among quantized candidates
```

The roles may intentionally differ. A fragile candidate can be best nominally
while a slightly worse nominal candidate is best under perturbation. This
distinction is part of the result, not a hidden tie-breaker.

## Reports and physics separation

`ROBUSTNESS_REPORT.json` contains the robustness settings, selected roles,
candidate certificate IDs/statuses, and formal robustness records. It carries
`physics_validity_is_separate: true` and never becomes a physics acceptance
certificate. `PHYSICS_ACCEPTANCE_CERTIFICATE.json` remains the authority for
nominal TMM validity.

No robust role can admit an unsupported TMM problem, self-sign a certificate,
or substitute an external solver. The same material coverage, scalar isotropic
planar capability, and nominal verifier gates remain in force.

