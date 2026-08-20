# Experiments and run state (v0.3)

VeriTMM v0.3 adds a local-first `ExperimentStore` around the existing
deterministic TMM execution path. The store is an index, not a second physics
engine: SQLite records identity and provenance, while run artifacts remain
ordinary files that can be inspected or archived.

## Store layout

By default, commands use the current working directory:

```text
.veritmm/
  experiments.db
  runs/
    <run_id>/
      RUN_RESULT.json
      RESULT_SUMMARY.json
      NORMALIZED_TASK.json
      PHYSICS_ACCEPTANCE_CERTIFICATE.json
      ...
```

Pass `--store-dir` to place the store elsewhere. A run directory is keyed by a
new invocation ID such as `run_<uuid>`. The SQLite record includes the
experiment and parent IDs, normalized task hash, execution identity, operation
and status, protocol/package versions, certificate and artifact roots, cache
provenance, tags, hypothesis, change reason, and caller metadata.

## Identity and metadata

`task_sha256` is computed from the normalized operation and task. Reordering
JSON object keys does not change the hash; changing a physical task does. Each
invocation still receives a new `run_id`, including cache hits.

Cache eligibility uses a separate execution identity that includes:

- the normalized task;
- package and protocol versions;
- the material-catalog identity; and
- numerical execution settings.

`hypothesis`, `change_reason`, tags, and `user_metadata` are research
provenance only. They are persisted in the store and deliberately do not enter
the physics acceptance certificate or the physics decision.

## Python API

```python
from tmm_engine.experiment_store import ExperimentStore, compare_runs

store = ExperimentStore(".veritmm")
record = store.record_envelope(
    run_result,
    artifact_root="output/run",
    experiment_id="exp_demo",
    hypothesis="Baseline coating",
    change_reason="Initial measurement model",
    user_metadata={"notebook": "A-01"},
)

store.get_run(record.run_id)
store.list_runs(experiment_id="exp_demo")
store.list_children(record.run_id)
store.get_lineage(record.run_id)
compare_runs(store, run_a, run_b)
```

The lineage and compare payloads are versioned machine-readable objects.
Compare reports deterministic task, material, solver, summary, certificate,
and artifact deltas; it does not rank designs or make a natural-language
scientific preference judgment.

## CLI inspection

```bash
veritmm history --store-dir .veritmm --json
veritmm inspect run_<id> --store-dir .veritmm --json
veritmm lineage run_<id> --store-dir .veritmm --json
veritmm compare run_<a> run_<b> --store-dir .veritmm --json
```

Each command emits one JSON object on stdout. Diagnostics belong on stderr.

## Cache provenance

A valid cache hit copies immutable artifacts into a new run directory and
records both `cache_hit: true` and `source_run_id`. The new run remains
independently queryable, and its summary/result artifact hashes are refreshed
after the new run ID and provenance fields are written. A changed task,
material catalog identity, protocol/package version, or execution setting
invalidates the cache.

## Physics boundary

Experiment metadata cannot widen the TMM capability boundary. Lateral periodic,
arbitrary-dimensional, anisotropic, non-plane-wave, and time-domain requests
remain typed preflight rejections. The store never routes them to RCWA, FDTD,
FEM, or another external solver, and it never converts a rejected task into a
physics certificate.

