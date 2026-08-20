# VeriTMM v0.5 implementation report

Date: 2026-08-20  
Status: v0.5.0 release accepted for repository publication on 2026-08-20

## Scope

v0.5 adds an evaluation and distribution layer around the existing verifier-
first TMM environment. It does not add a new Maxwell solver, place an LLM in the
physics core, or permit an optimizer or benchmark to issue a physics
certificate.

Implemented public components:

- `tmm_engine.agent_bench`: strict case contracts and deterministic offline
  execution;
- `tmm_engine.agent_harness`: framework-neutral traditional/agent-native A/B
  exposures, trajectory format, independent scoring, and callable runners;
- `veritmm benchmark --offline --json`;
- `veritmm agent-benchmark --trajectories ... --json`;
- packaged benchmark cases, controlled prompts, scoring policy, and a blind
  sample trajectory;
- machine-first entry documents (`AGENTS.md`, `CLAUDE.md`, and
  `docs/AGENTBENCH.md`); and
- v0.5 capability-manifest declarations, package data, README, architecture,
  validation, Agent Skill, and changelog updates.

The optional MCP adapter remains deliberately deferred. The CLI and Python
callable surfaces are complete without MCP, and no transport-specific physics
rule is duplicated.

## AgentBench catalogue

The maintained generator is `scripts/generate_agentbench_cases.py`. It writes
one strict `veritmm-agentbench-case-v1` JSON file per case and aborts if an
independent preflight audit disagrees with a declared failure code.

Final catalogue composition:

| Dimension | Count |
|---|---:|
| Total | 82 |
| Supported | 50 |
| Unsupported physics/output combinations | 18 |
| Invalid contracts or unresolved material inputs | 14 |
| Simulation | 60 |
| Optimization | 8 |
| Sweep | 5 |
| Sensitivity | 4 |
| Tolerance | 5 |
| Cases with real numerical execution | 22 |
| Cache replay scenarios | 1 |
| Sweep resume scenarios | 1 |

The executed subset includes ordinary and absorbing films, DBRs, mixed
coherence, ellipsometry, amplitudes, phase dispersion, layer absorption,
emissivity, optimization, robust optimization, sweeps, near-zero sensitivity,
and low-yield tolerance. Boundary cases include lateral/arbitrary geometry,
anisotropic/magneto-optic/nonlinear material declarations, non-plane-wave and
time-domain excitation, incompatible outputs, unresolved material data, and
invalid study contracts.

## Offline release result

Evidence:
`outputs/agentbench_v0_5_final/BENCHMARK_RESULT.json`

| Metric | Result |
|---|---:|
| Cases passed | 82 / 82 |
| Valid-case pass rate | 1.0 |
| Invalid-case rejection rate | 1.0 |
| Exact failure-code accuracy | 1.0 |
| Artifact completeness | 1.0 |
| Certificate success | 1.0 |
| Reproducibility | 1.0 |
| Unsupported false accepts | 0 / 18 |
| Unsupported false-accept rate | **0.0** |

Catalogue SHA-256:
`ce6dd3d44a330fe8146f1b9f21d2c502861f9fac73b7fdab80a370002362e0c6`

The benchmark ran without a network or LLM. It removes invocation identity and
timing from repeated-run fingerprints while retaining scientific/numerical
content. Cache replay and sweep resume are physically executed rather than
inferred from field presence.

## Blind-agent usability evidence

A fresh Luna agent received no conversation context and was forbidden to read
source, tests, case answers, generator code, or task examples. It read only the
machine entry documents and public command/schema surface, then independently:

1. discovered capabilities;
2. exported schemas;
3. made explicit assumptions for an underspecified six-period DBR;
4. constructed a valid task on its first attempt;
5. passed preflight;
6. completed the numerical run; and
7. received convergence, cross-solver, and physics acceptance.

Independent trajectory scoring recomputed preflight, normalized task identity,
operation, run-envelope schema, and certificate consistency. Results:

- first-valid-task rate: 1.0;
- final success rate: 1.0;
- correction turns: 0;
- certified success rate: 1.0;
- unsupported false-accept rate: 0.0;
- unavailable token/timing fields: `null`.

Evidence:

- `benchmarks/trajectories/blind_luna_dbr_v0_5.json`;
- `outputs/blind_agent_v0_5/AGENT_BENCHMARK_RESULT.json`; and
- `outputs/blind_agent_v0_5/RUN_SIM_DBR_SIX_PERIOD/`.

This is one usability trajectory, not a statistically powered claim that one
agent exposure is universally superior.

## Defects exposed by blind use

The blind test identified two protocol defects that ordinary regression tests
had missed:

1. A documented single `AgentTrajectory` object was silently interpreted as an
   empty collection. The loader now accepts one object, a non-empty list, or a
   non-empty wrapper and rejects empty evidence.
2. Numerically equivalent JSON such as `2` and `2.0` could produce different
   normalized task hashes. The task input adapters now coerce physical values
   to stable numeric types before identity, cache, lineage, or trajectory
   auditing.

The scorer was also hardened so a self-reported successful run cannot pass with
an invalid envelope or a task-hash mismatch.

## Distribution acceptance

The final wheel and source distribution build successfully. Wheel inspection
confirmed:

- all 82 cases;
- the blind trajectory;
- benchmark prompts and policy documents;
- the bundled refractiveindex.info database; and
- all engine and protocol modules.

The wheel was installed with `--no-deps` into a fresh Python 3.11 virtual
environment using system scientific dependencies. From outside the source
root, the installed package successfully executed `describe`, schema export,
preflight, all 82 packaged benchmark cases, and independent scoring of the
packaged blind trajectory.

Final wheel:
`outputs/package_build_v0_5_release_final/dist/veritmm-0.5.0-py3-none-any.whl`

## Integrity boundaries

- Optimizers propose; independent TMM recomputation and validation certify.
- Robustness evidence cannot change nominal physics validity.
- Material extrapolation remains fail-closed and is never enabled by a repair
  action.
- Out-of-domain physics is rejected with typed failures; no external solver is
  run or certified.
- Benchmark and agent evidence do not enter a certificate payload or identity.
- Missing agent metrics remain `null`; they are not estimated.

## Known limitations and deferred work

This is an alpha release candidate, not a claim of full production readiness.
Agent A/B statistics need more models, tasks, and repetitions before drawing
broad conclusions. The current solver boundary remains passive isotropic planar
TMM. Optional MCP transport, remote execution, multi-solver evidence graphs,
and RCWA/FDTD handoff are future work and are not implemented or claimed here.
