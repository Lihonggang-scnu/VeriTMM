# Artifact reference: what a run directory contains

A completed simulate run writes (in the run directory, or retrievable as
`veritmm://run/{run_id}/{name}` resources from the MCP surface):

| File | Kind | Contents |
|---|---|---|
| `RUN_RESULT.json` | run envelope | First-read entry: `ok`, `status`, `run_id`, `task_sha256`, `certificate_id`, `artifacts` index, `next_machine_actions` |
| `RESULT_SUMMARY.json` | result summary | Compact physics and spectral features |
| `NORMALIZED_TASK.json` | normalized task | The task after validation/normalization |
| `PREFLIGHT_REPORT.json` | preflight report | Contract/capability/material-coverage decision |
| `SIMULATION_RESULT.json` | simulation result | Full solver output: channels, extras, audit, raw material provenance |
| `SPECTRA.csv` / `SPECTRA.png` | spectrum table/plot | Per-channel spectra |
| `PHYSICS_ACCEPTANCE_CERTIFICATE.json` | physics certificate | The verifier's decision and its inputs |
| `VERIFICATION_EVIDENCE.json` | verification evidence | Raw decision inputs (facts) |
| `VERIFICATION_POLICY.json` | verification policy | The judgement rules actually applied |
| `RUN_MANIFEST.json` | legacy manifest | Execution metadata |

## Reading order

1. `RUN_RESULT.json` — status and identity first.
2. `RESULT_SUMMARY.json` — physics features without ingesting spectra.
3. `PHYSICS_ACCEPTANCE_CERTIFICATE.json` — only when the next decision needs
   the verdict detail.
4. `VERIFICATION_EVIDENCE.json` / `VERIFICATION_POLICY.json` — only when
   auditing how the verdict was reached.
5. `SPECTRA.csv` — only when spectra themselves are the question.

## Integrity model

Every artifact index entry carries `sha256` and `size_bytes`. The evidence
binds `schema_version`, `identity_scheme`, `run_id`, and `task_sha256`.
Never edit, move, or re-hash files inside a run directory — `verify-run`
re-computes all of it, and any mismatch fails integrity with typed codes.
