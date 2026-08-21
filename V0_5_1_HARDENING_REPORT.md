# VeriTMM v0.5.1 Scientific Semantics and CI Hardening Report

## Scope and outcome

This change set hardens v0.5.0 without adding a solver family, an LLM runtime,
MCP, or a user interface. It preserves the verifier-first boundary:

```text
PyTorch differentiable optimizer proposes
  -> independent NumPy TMM recomputes
  -> deterministic physics verifier decides
  -> physics acceptance certificate records the decision
```

The local release candidate passes the complete source regression suite, the
real PyTorch test subset, the offline AgentBench release gate, source/wheel
builds, and an isolated-wheel capability/preflight smoke test.

## Scientific semantic changes

### Append-only run provenance

- `ExperimentStore.record_run()` now uses an ordinary SQLite `INSERT`.
- Reusing a `run_id` raises `RunLedgerConflictError`; the machine contract maps
  it to `provenance_conflict` with a safe `use_fresh_run_identity` action.
- Canonical run directories and cache destinations must be empty. Existing
  non-empty artifact trees are not merged or overwritten.
- Cache replay still creates a fresh invocation `run_id`.
- Cached sweep children also receive fresh invocation IDs, rewritten child
  artifacts and CSV references, updated `SWEEP_RESULT` references, and an
  atomically appended parent/child ledger group linked to the source runs.
- `update_run_status()` is the only explicit lifecycle mutator and can update
  only `status`, `completed_at`, and `certificate_id`.

### Tolerance result v2

`veritmm-tolerance-result-v2` and `veritmm-robustness-report-v2` distinguish:

```text
conditional_yield = target_pass_count / completed_sample_count
overall_success_fraction = target_pass_count / requested_sample_count
```

Wilson intervals use `completed_sample_count`. Computational failures are
classified as `invalid_perturbed_design`, `numerical_failure`,
`material_failure`, or `unexpected_runtime_failure`. An all-failed ensemble has
null conditional yield, interval, and descriptive statistics, plus the typed
failure `insufficient_valid_samples`.

A returned solver object is not automatically a completed sample. Non-finite
observables, passivity-bound violations, and energy-audit failures are rejected
before metric or objective accounting.

### Robust selection without survivor bias

A candidate can win `best_robust` only when nominal independent validation and
physics acceptance pass and the final robustness ensemble is complete with zero
failed samples. Incomplete candidates receive `robust_objective: null` and
`eligible_for_robust_selection: false`; a nominally valid one may still remain
`best_nominal`.
The compatibility alias `most_robust` is overwritten with `best_robust`, so the
pre-final heuristic cannot bypass this gate.
If the formal evaluator itself fails, every robust winner role remains null and
the nominal portfolio records `formal_evaluation_failed`; no heuristic fallback
is permitted.

### Shared uncertainty policy

Robust training and independent final evaluation now declare and use the same:

- distribution: `normal`;
- boundary policy: `truncate`;
- minimum physical thickness: 0.1 nm by default; and
- deterministic offset generator.

Training uses the declared training seed. Final proof uses a separate,
deterministic seed. Both seeds and the policy are serialized. Tolerance analysis
uses the same boundary implementation while retaining its per-layer normal or
uniform distribution contract.

## Schemas and compatibility

### Changed public output schemas

- `veritmm-tolerance-result-v1` -> `veritmm-tolerance-result-v2`
- `veritmm-robustness-report-v1` -> `veritmm-robustness-report-v2`
- `veritmm-result-summary-v1` -> `veritmm-result-summary-v2`

### Additive input fields

Old optimization and tolerance tasks remain valid because new fields have
defaults:

```json
{
  "distribution": "normal",
  "boundary_policy": "truncate",
  "min_thickness_physical_nm": 0.1
}
```

The robustness contract supports only `normal`; declaring another distribution
fails validation rather than silently changing semantics.

### Yield migration note

In tolerance v2, compatibility fields `yield`, `yield_ci95`, and
`target_pass_probability` are aliases of conditional yield over completed
samples. A consumer that needs the former requested-sample denominator must use
`overall_success_fraction`. The schema version prevents this semantic change
from being silent.

Solver inputs, the Python simulation API, `run_tmm_task.py`, protocol version,
and the physics acceptance certificate schema are unchanged.

## Modified files

### Runtime and protocol

- `tmm_engine/experiment_store.py`
- `tmm_engine/managed_execution.py`
- `tmm_engine/execution.py`
- `tmm_engine/capabilities.py`
- `tmm_engine/uncertainty.py` (new)
- `tmm_engine/scientific_analysis.py`
- `tmm_engine/robust_optimization.py`
- `tmm_engine/optimization.py`
- `tmm_engine/protocol/models.py`
- `tmm_engine/protocol/capabilities.py`
- `tmm_engine/schemas.py`
- `tmm_engine/task_io.py`
- `tmm_engine/run_artifacts.py`
- `tmm_engine/__init__.py`

### Tests, CI, packaging, and documentation

- `.github/workflows/ci.yml`
- `MANIFEST.in`
- `pyproject.toml`
- `tests/test_capability_manifest.py`
- `tests/test_experiment_store.py`
- `tests/test_run_cache.py`
- `tests/test_yield_statistics.py`
- `tests/test_uncertainty_sampling.py`
- `tests/test_robustness_report.py`
- `tests/test_robust_optimization.py`
- `tests/test_robust_uncertainty_policy.py` (new)
- `tests/test_sensitivity_autodiff.py`
- `tests/test_sensitivity_finite_difference.py`
- `tests/test_tmm_differentiable.py`
- `README.md`
- `README.zh-CN.md`
- `CHANGELOG.md`
- `docs/EXPERIMENTS.md`
- `docs/UNCERTAINTY.md`
- `docs/ROBUST_OPTIMIZATION.md`
- `docs/VALIDATION.md`

## Verification evidence

Baseline before hardening: 191 tests passed and Ruff passed.

Local release gate:

| Gate | Result |
|---|---:|
| `python -m pytest -q` | 214 passed, 0 skipped |
| `python -m ruff check .` | passed |
| `pytest -m requires_torch` | 14 passed, 0 skipped, 200 deselected |
| robust optimization/policy subset | 9 passed |
| `python -m build` | sdist and wheel built |
| isolated v0.5.1 wheel `describe --json` | passed |
| isolated v0.5.1 wheel preflight | passed |
| offline AgentBench | 82/82 passed |
| AgentBench unsupported false acceptance | 0/18 = 0.0 |

Built artifacts:

```text
dist/veritmm-0.5.1-py3-none-any.whl
dist/veritmm-0.5.1.tar.gz
```

The controlled tolerance test verifies 10 requested, 8 completed, 6 passed,
and 2 computationally failed samples. The survivor-bias test verifies that a
nominally better candidate with 20% final-sample failures cannot beat a fully
evaluated robust candidate.
The cache adversarial test verifies that replaying a two-child sweep creates a
new parent and two new child identities, while every new child ledger row points
back to its original source child. The robust fallback test verifies that a
failed formal evaluator leaves no robust winner alias.

An independent read-only Luna Max audit initially blocked release on four hidden
edge cases: invalid returned spectra counted as completed, unversioned summary
yield semantics, stale run IDs in copied cache artifacts, and the legacy
`most_robust` alias bypassing the formal gate. Each finding was corrected and
covered by an adversarial regression. A second pass then found stale sweep CSV
IDs, overly broad child paths, non-atomic parent/child ledger writes, a
robustness-disabled heuristic alias, and a post-write summary hazard. Those were
also corrected before the final release gate was rerun.
The final narrow audit confirmed fail-closed sweep-cache validation, including
successful child ledger state, matching child identity, and `ok=true`, as well
as rejection of output-tree links and Windows reparse points.

## CI jobs

- `core`: Python 3.10, 3.11, and 3.12 source regression.
- `torch`: installs the optimization extra, imports PyTorch, runs the
  `requires_torch` marker, machine-checks JUnit for zero skipped tests, and runs
  robust portfolio/report tests.
- `lint`: Ruff over the complete repository.
- `agentbench`: installs the optimization extra, executes the offline benchmark,
  and independently checks case counts, release status, and zero unsupported
  false acceptance.
- `build`: builds sdist/wheel, installs the wheel in a fresh environment, then
  parses machine capability JSON.

## Remaining limitations

- `truncate` is the only public robustness boundary policy in v0.5.1. A future
  policy requires a new explicit contract and matched Torch/NumPy tests.
- Final robust selection deliberately uses a strict zero-computational-failure
  rule. Failure-tolerant statistical estimators are not implemented.
- Artifact copies use private staging directories and atomic rename. A hard
  process kill can leave an unindexed hidden `.staging` directory, but it never
  becomes a canonical run or ledger row and can be removed during maintenance.
- The core CI job may skip optional Torch modules because it installs only the
  core test extra. The dedicated Torch job is the release authority for those
  modules and requires zero skips.
- The protocol remains scalar, passive, isotropic, planar TMM. No change in this
  release expands the documented physics domain.

## Release recommendation

The implementation is suitable as a local v0.5.1 release candidate after the
independent review reported for this working tree. No commit, push, or tag was
performed by this task.
