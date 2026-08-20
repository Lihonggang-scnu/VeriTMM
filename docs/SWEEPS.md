# Formal parameter sweeps (v0.3)

Sweeps are finite, versioned `sweep-task-v1` studies over a validated base
simulation. They are data contracts, not arbitrary Python callbacks.

## Task shape

```json
{
  "schema_version": "sweep-task-v1",
  "mode": "sweep",
  "sweep": {
    "base_simulation": {"stack": {}, "spectrum": {}},
    "parameters": [
      {"path": "/stack/layers/0/thickness_nm", "values": [80, 90, 100]}
    ],
    "metrics": [
      {
        "name": "mean_R",
        "observable": "R",
        "wavelength_min_nm": 500,
        "wavelength_max_nm": 600,
        "aggregation": "mean",
        "angle_deg": 0,
        "polarization": "unpolarized"
      }
    ]
  }
}
```

The allow-list currently permits JSON-pointer changes to:

- `/stack/layers/<index>/thickness_nm`;
- `/illumination/angles_deg/<index>`;
- `/spectrum/start_nm`;
- `/spectrum/stop_nm`; and
- `/spectrum/points`.

Material identity, material datasets, solver selection, requested outputs, and
physics declarations are not sweep axes. This prevents a sweep from silently
changing the scientific model or bypassing capability validation.

## Ordering and metrics

Axes are expanded in declaration order using the values in the order supplied
by the caller. Cartesian rows therefore have stable integer indices and stable
child task hashes. Supported reductions are `mean`, `min`, `max`, `worst_case`,
`value_at_wavelength`, and `threshold_band_width`. Observables are `R`, `T`,
`A`, and `E_system` when that observable is actually emitted with valid
semantics by the selected simulation.

Every child is an independent simulation task with its own normalized task,
task hash, run ID, status, metrics, failures, and artifacts. A failed child is
retained in the study result; it is never silently removed or converted to a
successful metric row. A study can report partial success while its parent
`ok` value remains false.

## Artifacts

The parent output contains:

```text
SWEEP_RESULT.json
SWEEP_TABLE.csv
RUN_RESULT.json
RESULT_SUMMARY.json
NORMALIZED_TASK.json
RUN_MANIFEST.json
children/<index>_<hash-prefix>/
  RUN_RESULT.json
  RESULT_SUMMARY.json
  NORMALIZED_TASK.json
  PHYSICS_ACCEPTANCE_CERTIFICATE.json  # successful child
  SPECTRA.csv                           # successful child
```

`SWEEP_RESULT.json` and `SWEEP_TABLE.csv` use versioned artifact identifiers.
When a store is supplied, the parent is the lineage root and successful or
failed child envelopes are archived below it as `sweep_child` records.

## Resume

Execution checkpoints the study after every child. Resume reads the existing
study task hash and reuses only children that are complete and whose child
task hash still matches. Pending and failed children can be attempted again;
completed children keep their original child run ID and artifacts. A changed
study task hash is rejected rather than producing a mixed study. Resume keeps
the parent run ID so the study remains one auditable invocation.

## Fail-closed behavior

Sweep expansion does not grant new physics capability. Each child is parsed and
validated through the normal TMM preflight/runtime path. Unsupported geometry,
material models, excitation, time-domain requests, invalid material coverage,
and invalid task values remain typed failures. No external solver handoff is
performed.

