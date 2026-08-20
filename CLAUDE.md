# VeriTMM coding-agent entry point

Read and follow [`AGENTS.md`](AGENTS.md) before editing or running a task.

The shortest safe workflow is:

```bash
veritmm describe --json
veritmm schema simulation
veritmm preflight TASK.json --json
veritmm run TASK.json --output-dir RUN_DIR --json
```

Do not inspect the numerical kernel to guess a JSON contract. Do not relax
physics validation, extrapolate material data silently, or coerce out-of-scope
physics into scalar planar TMM. `RUN_RESULT.json` is the machine-readable entry
point; the optimizer, benchmark, and agent are never certificate authorities.
