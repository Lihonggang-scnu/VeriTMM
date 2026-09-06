# Protocol reference: tool surface and task format

## Identity

Every run declares `identity_scheme: veritmm-canonical-json-v1` — the
canonical-JSON convention (UTF-8, sorted keys, compact separators,
`ensure_ascii=False`, `allow_nan=False`) behind `task_sha256`,
`certificate_id`, and cache identity.

## Tool surface (MCP) and CLI fallback

| Capability | MCP tool | CLI fallback |
|---|---|---|
| Capability manifest | `describe` | `veritmm describe --json` |
| Capability catalog (runtime) | `capability-catalog` | `python -m tmm_engine.cli capability-catalog --check` |
| Task contract | `schema` (kind=simulation) | `veritmm schema simulation` |
| Example tasks | `examples` | `veritmm examples --json` |
| Gate without running | `preflight` | `veritmm preflight TASK.json --json` |
| Execute one simulate task | `run` | `veritmm run TASK.json --output-dir DIR --json` |
| Persisted runs | `history` | `veritmm history --json` |
| One run | `inspect` | `veritmm inspect RUN_ID --json` |
| Ancestry | `lineage` | `veritmm lineage RUN_ID --json` |
| Diff two runs | `compare` | `veritmm compare A B --json` |
| Independent verification | `verify_run` | `veritmm verify-run RUN_DIR --json` |

Operator/debug parameters (`skip_certificate`, `physics_python`, convergence
tolerances, `device`, `output_dir`, `store_dir`) do not exist on the MCP
surface. Rejections arrive either at the SDK schema level or as a typed
`McpToolError` — the guarantee is that the parameter is unavailable.

## Task format (simulate)

```json
{
  "mode": "simulate",
  "simulation": {
    "stack": {
      "layers": [
        {"material": "tio2", "thickness_nm": 60.0},
        {"material": "sio2", "thickness_nm": 90.0}
      ],
      "incident": {"material": null, "constant_n": 1.0},
      "exit": {"material": "sio2"}
    },
    "spectrum": {"start_nm": 500.0, "stop_nm": 600.0, "points": 11},
    "illumination": {"angles_deg": [0.0], "polarizations": ["s"]},
    "solver": "smatrix",
    "requested_outputs": ["R", "T", "A"]
  }
}
```

- Wavelengths and thicknesses: **nanometres**. Angles: **degrees**.
- Units are never inferred; a task missing required fields fails preflight.
- Material selection: set `material` (canonical name resolved through the
  registry) or `constant_n`/`constant_k`. Dataset identity (`provider`,
  `dataset_id`) should be stated when reproducing results.

## Responses

Tool responses are compact envelopes: `ok`, `status`, `run_id`,
`task_sha256`, `certificate_id`, and an `artifacts` index whose entries carry
`kind`, `path`, `schema_version`, `sha256`, `size_bytes`. Response profiles:
`compact` (default), `standard`, `full`; arrays always live in artifacts,
never in the response.

MCP resources: `veritmm://task/{path}` (server task root) and
`veritmm://run/{run_id}/{path}` (one artifact of a server-executed run).
