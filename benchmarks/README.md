# VeriTMM-AgentBench v1

This directory is the versioned, offline evaluation set for VeriTMM's public
agent protocol. It tests task construction, preflight routing, typed failures,
execution artifacts, physics assertions, cache/restart behavior, and
reproducibility. It never changes the physics acceptance path.

## Layout

- `cases/`: one strict JSON contract per benchmark case;
- `prompts/`: controlled traditional and agent-native exposure instructions;
- `trajectories/`: recorded external-agent evidence;
- `expected/` and `scoring/`: notes about reference semantics and scoring;
- `scripts/generate_agentbench_cases.py`: deterministic catalogue generator.

The release catalogue contains at least 80 cases spanning simulation,
optimization, sweep, sensitivity, tolerance, unsupported physics, incompatible
outputs, invalid contracts, and unresolved material data. A meaningful subset
executes the numerical engine twice. Cache replay and sweep resume are executed,
not inferred from schema acceptance.

Run the release gate:

```bash
veritmm benchmark --offline --json
```

The command makes no network or LLM calls. Passing requires every case to meet
its declared outcome and `unsupported_false_accept_rate == 0`.

Regenerate after intentionally editing the maintained catalogue definition:

```bash
python scripts/generate_agentbench_cases.py
```

Generation independently preflights every declaration and aborts on mismatch.
Do not derive expected failures dynamically from benchmark execution.

See `docs/AGENTBENCH.md` for the trajectory format and A/B harness.
