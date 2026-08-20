# VeriTMM v0.3 to v0.5 final local acceptance

Date: 2026-08-20  
Decision: release-candidate gates passed locally; awaiting user approval before
any Git commit/tag/push or public release

## Delivered progression

### v0.3 — agent-native experimentation

- SQLite/filesystem `ExperimentStore`;
- immutable normalized task identity and unique invocation identity;
- experiment/run lineage, history, inspect, and deterministic compare;
- provenance-preserving cache replay;
- finite allow-listed sweeps, child artifacts, partial-failure records, and
  checkpoint resume.

### v0.4 — scientific analysis

- thickness sensitivity with independent NumPy central-difference audit;
- seeded normal/uniform tolerance sampling;
- failed-sample accounting, yield, percentiles, and Wilson confidence interval;
- separate robustness reports;
- expected, worst-case, and mean-plus-sigma robust optimization objectives;
- independently evaluated nominal/robust/quantized candidate roles.

### v0.5 — evaluation and distribution

- 82-case deterministic AgentBench;
- framework-neutral external-agent trajectory and A/B harness;
- independent trajectory task/run/certificate audit;
- blind unfamiliar-agent usability run;
- machine-first bootstrap documents;
- installable benchmark, prompts, policies, and sample trajectory;
- package version `0.5.0`.

## Final acceptance matrix

| Gate | Result |
|---|---|
| Ruff | passed |
| Full pytest | **191 passed** |
| Bundled real examples | **7 / 7 completed** |
| Offline AgentBench | **82 / 82 passed** |
| Unsupported false acceptance | **0 / 18 (0.0)** |
| Exact typed failure codes | **32 / 32** |
| Repeated scientific content | **82 / 82 reproducible** |
| Expected artifact sets | **22 / 22 complete** |
| Expected physics certificates | **15 / 15 accepted** |
| Blind agent first task | valid |
| Blind agent certified run | passed |
| Source distribution build | passed |
| Wheel build | passed |
| Fresh wheel installation | passed |
| Packaged benchmark | 82 / 82 passed |
| Packaged trajectory audit | passed |

All acceptance runs were local. The deterministic suite and AgentBench made no
network or LLM calls. The one external blind-agent trajectory was evaluation
evidence only and did not participate in physics certification.

## Backward compatibility

The existing simulation/optimization JSON forms, Python dataclass API,
`scripts/run_tmm_task.py`, material registry, forward solvers, and legacy
artifact names remain available. New state and study features are additive.
Numeric JSON normalization was strengthened so semantically equal integer and
floating representations now share task identity; this intentionally corrects
an earlier cache/provenance ambiguity.

Machine-mode success and failure paths remain single JSON objects on stdout;
human/debug output remains separate. `RUN_RESULT.json` remains the unified
artifact index and `RESULT_SUMMARY.json` the compact first-read result.

## Scientific acceptance statement

The release candidate demonstrates a bounded, reproducible TMM environment for
passive isotropic planar multilayers. It does not claim that TMM is sufficient
for lateral gratings, metasurfaces, anisotropy, arbitrary 2D/3D geometry,
nonlinear optics, finite beams, dipoles, or time-domain fields. Those requests
fail closed with structured guidance.

No optimizer, research metadata record, benchmark, or agent can sign a physics
certificate. Robustness and tolerance claims remain separate from nominal
physical validity. Material substitutions and extrapolation require explicit
scientific choice.

## Primary evidence paths

- `outputs/agentbench_v0_5_final/BENCHMARK_RESULT.json`
- `outputs/blind_agent_v0_5/AGENT_BENCHMARK_RESULT.json`
- `outputs/blind_agent_v0_5/RUN_SIM_DBR_SIX_PERIOD/`
- `outputs/example_release_smoke_20260820/`
- `outputs/package_build_v0_5_release_final/dist/`
- `outputs/package_install_v0_5_release_final/`
- `V0_3_IMPLEMENTATION_REPORT.md`
- `V0_4_IMPLEMENTATION_REPORT.md`
- `V0_5_IMPLEMENTATION_REPORT.md`

## Deferred, not hidden

- larger multi-model A/B studies;
- optional thin MCP transport;
- remote/HPC execution;
- RCWA/FDTD/FEM handoff and cross-solver evidence graphs;
- broader material-uncertainty models beyond thickness;
- production CI publishing/signing policy.

These items are outside the accepted v0.5 scope. The current CLI and Python
protocol are complete without them.
