# Scoring policy

Offline release metrics are computed by `tmm_engine.agent_bench`:

- valid-case pass rate;
- invalid-case rejection rate;
- exact expected failure-code accuracy;
- artifact completeness;
- certificate success;
- repeated-run reproducibility;
- unsupported-physics false-accept rate.

All release cases must pass, the catalogue must contain at least 80 cases, and
the unsupported false-accept rate must equal zero.

Agent A/B metrics are computed independently from task attempts and final run
envelopes where available. Missing token, timing, or reproducibility evidence is
reported as `null`, never estimated or invented.
