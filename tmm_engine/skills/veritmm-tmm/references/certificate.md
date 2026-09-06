# Certificate and the four verify-run states

## PHYSICS_ACCEPTANCE_CERTIFICATE.json

Key fields:

- `task_sha256` / `task_hash_scope` — the verified task identity (raw-task
  scope; the run envelope carries the normalized-operation scope).
- `capability_assessment` — engine scope decision with typed failures.
- `physics_audit` — raw audit quantities: `energy_conservation_max_abs_error`,
  `nonfinite_value_count`, `passivity_check_passed`, worst-case locations.
- `spectral_convergence` — convergence ledger (`status`, `passed`, rounds,
  declared vs verification grid hashes).
- `independent_solver_check` — primary vs reference solver agreement.
- `tightest_margin` — the acceptance check closest to its threshold.
- `high_precision_referee` — informational mpmath referee report; it never
  changes `accepted`.
- `material_provenance_sha256`, `material_catalog` — input data identity.
- `accepted`, `status`, `failures`, `evidence_coverage`, `certificate_id` —
  the verdict. `status` is `physically_valid`,
  `physically_valid_with_limits`, or `rejected_physics`.

## The four verify-run states (`veritmm verify-run` / `verify_run` tool)

| State | Values | Meaning |
|---|---|---|
| `integrity_status` | `valid` / `invalid` | Every indexed artifact matches its recorded hash and size; evidence binding (schema_version, identity_scheme, run_id, task_sha256) holds |
| `certification_status` | `certified` / `uncertified` / `not_evaluated` | `certified`: a full acceptance certificate was issued (convergence and independent-solver checks ran). `uncertified`: never fully certified (e.g. both checks absent — the skip-certificate signature) |
| `replay_status` | `passed` / `failed` / `unavailable` / `not_evaluated` | `passed`: `evaluate_evidence()` on the persisted evidence+policy reproduces the certificate. `unavailable`: cache replays carry a certificate from an earlier run and no evidence of their own; legacy v1.0 runs may have insufficient evidence |
| `authenticity_status` | `unsigned` (1.1) | Detached DSSE signatures arrive in 1.2 |

`overall_status` is the machine summary: `valid` (integrity valid +
certified + replay passed), `invalid` (integrity or replay failure), or
`incomplete` (trustworthy as far as checked, chain not fully closed).
CLI exit codes: 0 valid, 2 invalid, 3 incomplete.

## Interpretation rules

- The four states are independent. A cache replay is `certified` with
  `replay_status: unavailable` — normal, not a failure.
- An integrity failure stops everything: the remaining states are
  `not_evaluated` and the report carries typed failure codes
  (`sha256_mismatch`, `certificate_identity_mismatch`,
  `evidence_certificate_task_mismatch`, ...). Never interpret science from a
  run with `integrity_status: invalid`.
- The certificate proves nominal physics acceptance. It does **not** prove
  robustness, high yield, or fabrication validity — those are separate study
  artifacts.
