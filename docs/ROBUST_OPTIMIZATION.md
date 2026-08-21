# Robust optimization (v0.5.1)

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
    "distribution": "normal",
    "thickness_sigma_nm": 2.0,
    "k_sigma": 2.0,
    "boundary_policy": "truncate",
    "min_thickness_physical_nm": 0.1
  }
}
```

Supported formal objectives are:

- `expected_loss` — mean perturbed loss;
- `worst_case_loss` — maximum completed perturbed loss; and
- `mean_plus_k_sigma` — mean plus the declared multiple of standard deviation.

Training and final evaluation use the same normal distribution and `truncate`
boundary semantics. Training uses the declared `seed`; final proof uses a
distinct deterministic `final_seed`, both serialized in the report. Final
robustness evaluation uses an independent NumPy S-matrix sampling path.
Training-time Monte Carlo only proposes designs and is never the final proof.
Failed perturbed samples are retained with typed failures.

## Candidate eligibility and roles

Only candidates with both `independent_validation_status: "passed"` and a
physics status of `physically_valid` or `physically_valid_with_limits` are
eligible for formal robustness evaluation. To win `best_robust`, the final
ensemble must additionally be complete, have zero failed samples, and have a
defined robust objective. Survivor-only statistics are diagnostic only: an
incomplete candidate receives `robust_objective: null` and
`eligible_for_robust_selection: false`. It may still be `best_nominal` if its
nominal independent validation passed. The evaluator exposes interpretable
roles rather than an opaque aggregate score:

```text
best_nominal   lowest nominal objective loss
best_robust    lowest formal robustness objective
best_quantized lowest formal robustness objective among quantized candidates
```

The deprecated `most_robust` portfolio alias is retained for compatibility but
is rewritten to exactly the same candidate as `best_robust`; it cannot preserve
the earlier heuristic winner and bypass the completeness gate.
When formal robustness is disabled, the cheap screening result is exposed only
as `best_heuristic_robustness`; `most_robust` remains null because no formal
robust claim was evaluated.

Before formal final evaluation completes, all robust roles are explicitly null.
If that evaluator raises or cannot finish, the nominal portfolio remains available
with `robust_selection_status: "formal_evaluation_failed"` and a typed diagnostic,
but `best_robust`, `best_quantized`, and `most_robust` remain null. This is a
fail-closed result, not permission to reuse the earlier heuristic ranking.

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
