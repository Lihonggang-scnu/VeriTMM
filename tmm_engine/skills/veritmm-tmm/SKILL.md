---
name: veritmm-tmm
description: Run verified multilayer-optics simulations through VeriTMM's verifier-first transfer-matrix protocol and receive a physics acceptance certificate. Use for thin films, DBRs, photonic crystals, defect cavities, chirped stacks, and absorbers — any passive, isotropic, planar 1-D multilayer task. Prefer the veritmm MCP tools; the veritmm CLI is the fallback. Every result is independently verifiable with verify_run.
---

# VeriTMM — verifier-first multilayer optics (TMM)

VeriTMM executes passive, isotropic, planar 1-D multilayer optics and issues a
machine-readable physics acceptance certificate for every result. The division
of labor is fixed: **the agent proposes, VeriTMM computes, the deterministic
verifier certifies.** Neither you nor any optimizer can certify a result —
`verify_run` re-derives the verdict from persisted evidence and policy.

## Preferred surface: the `veritmm` MCP tools

Servers start in **compact** mode: only `veritmm_discover` and
`veritmm_catalog_get` are bound; call `veritmm_discover` first and request the
full grid for the complete namespace.  Every tool description carries an
effect tag — `[READ_ONLY]`, `[COMPUTATIONAL]`, or `[EXPENSIVE_COMPUTE]`.

Namespaced grid (full mode; 15 tools, underscore style):

| Tool | Tag | Purpose |
|---|---|---|
| `veritmm_discover` | [READ_ONLY] | Compact discovery of the tool grid |
| `veritmm_catalog_get` | [READ_ONLY] | Runtime-generated capability catalog |
| `veritmm_material_search` / `veritmm_material_explain` | [READ_ONLY] | Bundled optical-constant catalog |
| `veritmm_stack_validate` | [READ_ONLY] | Validate a stack skeleton |
| `veritmm_simulate_spectrum` | [COMPUTATIONAL] | Managed simulate execution |
| `veritmm_simulate_batch` | [EXPENSIVE_COMPUTE] | Batch of isolated jobs + aggregate report |
| `veritmm_gradient_compute` | [COMPUTATIONAL] | Thickness gradients (proposal basis) |
| `veritmm_sensitivity_analyze` | [COMPUTATIONAL] | Normalized sensitivities (proposal basis) |
| `veritmm_optimize_problem` | [EXPENSIVE_COMPUTE] | Declarative problem -> optimize -> certify |
| `veritmm_verify_run` | [COMPUTATIONAL] | Independent four-state verification |
| `veritmm_verify_explain` | [COMPUTATIONAL] | Why is run A better than run B (provenance) |
| `veritmm_job_status` / `veritmm_job_result` | [READ_ONLY] | Job state and outputs |
| `veritmm_intent_compile` | [COMPUTATIONAL] | IntentSpec -> task + equivalence certificate |

Legacy grid (still bound in full mode; `run` and `verify_run` are deprecated
aliases of `veritmm_simulate_spectrum` / `veritmm_verify_run`):

| Tool | Purpose |
|---|---|
| `describe` | Capability manifest + protocol identity scheme |
| `schema` | Export the exact task JSON Schema (`kind`: simulation) |
| `examples` | List bundled example tasks |
| `preflight` | Gate one task without running a spectrum |
| `run` | Managed execution of one simulate task (default acceptance settings) |
| `history` / `inspect` / `lineage` / `compare` | Persisted experiment state |
| `verify_run` | Independent four-state verification of a run |

CLI fallback (same capabilities, same semantics): `veritmm <command> --json`.
Large artifacts are MCP resources (`veritmm://run/{run_id}/{path}`); with the
CLI they are files inside the run directory.

## Call order

1. `describe` — capability manifest and identity scheme.
2. `schema simulation` — read the contract **before** writing task JSON.
3. Compose ONE complete task: wavelengths and thicknesses in **nanometres**,
   angles in **degrees**; select material datasets explicitly when identity
   matters.
4. `preflight` — if rejected, follow the typed failure codes and safe actions;
   never guess a repair and never work around a rejection.
5. `run` — returns the compact envelope: `status`, `certificate_id`, artifact
   references. Fetch details via resources only when the next decision needs
   them.
6. `verify_run` — read the four states: `integrity_status`,
   `certification_status`, `replay_status`, `authenticity_status`, and
   `overall_status`. A cache replay is `certified` with `replay_status:
   unavailable` — that is normal and not a failure.

## Boundaries and red lines

- Supported physics: passive, isotropic, planar 1-D multilayers (thin films,
  DBRs, photonic crystals, defect cavities, chirped stacks, absorbers, finite
  substrates). Gratings, metasurfaces, anisotropy, nonlinear optics, finite
  beams, dipoles, mode sources, and time-domain requests **fail closed** with
  typed failures — do not attempt workarounds.
- `A = 1 - R - T` is absorption in finite layers; `E_system = 1 - R` is a
  different quantity with a different validity condition.
- A successful run or a high score is **not** physics validity: the verdict
  lives in the certificate and in `verify_run`.
- Robustness and yield are separate artifacts; a physics certificate does not
  imply robust optimality.
- Never modify or re-hash artifacts inside a run directory.
- `run` on this surface is simulate-mode only; optimization, sweeps,
  sensitivity, and tolerance studies are operator-side CLI work outside this
  skill.

Details: `references/protocol.md` (tool surface and task format),
`references/artifacts.md` (artifact sets and reading order),
`references/certificate.md` (certificate fields and the four verify-run
states), `references/boundaries.md` (capability boundary and material
governance).
