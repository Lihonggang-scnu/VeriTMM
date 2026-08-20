# VeriTMM-AgentBench and external-agent evaluation

## Purpose

AgentBench answers two separate questions:

1. Does the public protocol accept valid planar multilayer tasks and reject
   invalid or out-of-scope tasks deterministically?
2. Does an unfamiliar coding agent use VeriTMM more reliably when it receives
   capability discovery, schemas, preflight, typed actions, and result envelopes?

Neither answer modifies a physics certificate.

## Offline benchmark

Each strict `veritmm-agentbench-case-v1` record contains the natural-language
request, canonical task, expected mode/capability/failure codes, expected
artifacts, assertions, execution policy, and reproducibility count. Case files
reject unknown fields. Rejected cases stop at preflight. Executed cases run
twice; invocation identities and timing are removed before scientific-content
fingerprints are compared.

`cache_replay` executes a source run and a real cache hit. `sweep_resume`
executes a sweep and then resumes the same output directory, requiring completed
children to be reused.

```bash
veritmm benchmark --offline \
  --cases-dir benchmarks/cases \
  --output BENCHMARK_RESULT.json \
  --work-dir benchmark-work \
  --json
```

The result is versioned and records catalogue/content SHA-256 values. The
release gate requires at least 80 cases, all declared outcomes, and zero false
acceptance of unsupported physics.

## AgentTrajectory v1

External agents remain outside the core package. A runner records evidence:

```json
{
  "schema_version": "veritmm-agent-trajectory-v1",
  "benchmark_case": "sim_single_film_normal",
  "model": "external-model-name",
  "agent_version": "runner-version",
  "exposure": "agent_native",
  "prompt": "the exact controlled prompt",
  "steps": [{"index": 0, "action": "describe", "observation": "..."}],
  "tool_calls": [{"tool": "veritmm", "arguments": ["describe", "--json"]}],
  "correction_turns": 0,
  "final_run_id": "reported value",
  "certificate_id": "reported value or null",
  "success": true,
  "task_attempts": [{"mode": "simulate", "simulation": {}}],
  "final_run_result": {"schema_version": "veritmm-run-result-v1"},
  "input_tokens": null,
  "output_tokens": null,
  "wall_seconds": null
}
```

The scorer independently preflights `task_attempts`, validates the embedded run
envelope schema, recomputes the normalized final-task hash, checks operation and
certificate consistency, and only then reads run/certificate IDs. When the case
catalogue is supplied, it recomputes success and unsupported false acceptance
against the case contract. Self-reported success is not trusted. A single
trajectory object, a list, or a non-empty `{"trajectories": [...]}` wrapper is
accepted; empty evidence is rejected.

```bash
veritmm agent-benchmark \
  --trajectories benchmarks/trajectories/sample.json \
  --cases-dir benchmarks/cases \
  --output AGENT_AB_RESULT.json \
  --json
```

Unavailable metrics remain `null`. The framework-neutral Python entry points
are `build_exposure`, `run_agent_ab`, and `score_trajectories` in
`tmm_engine.agent_harness`.

## Controlled A/B interpretation

The traditional arm sees the README, Python API, and basic examples. The
agent-native arm additionally receives capability discovery, the relevant JSON
schema, preflight, typed actions, the result envelope, experiment state, and
study commands. Both arms receive the same natural-language case.

This benchmark measures protocol usability, not model intelligence. Report the
agent/model version, exact prompt, trajectory evidence, unavailable fields, and
sample size. Do not claim broad superiority from a single trajectory.

## Boundaries

No proprietary SDK is imported by VeriTMM. MCP remains optional and deferred;
the CLI and Python callable adapter are complete transports. Benchmark fixtures,
prompts, trajectories, and scoring are evidence assets, never inputs to the TMM
physics core.
