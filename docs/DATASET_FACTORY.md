# DatasetFactory

`DatasetFactory` creates deterministic, verified, artifact-backed research
datasets through this fixed path:

```text
DesignSpace -> SamplingPlan -> stable candidates
            -> ResearchEvaluator.evaluate_many()
            -> managed VeriTMM runs and certificates
            -> DATASET_INDEX.jsonl
```

It never invokes a solver, preflight implementation, or certifier directly.

## Sampling plans

`SamplingPlan` is immutable and content-addressed by `plan_id`.

| Strategy | Deterministic behavior |
|---|---|
| `random` | Stateless indexed sampling compatible with `DesignSpace` |
| `grid` | Declared-variable Cartesian order, bounded by `sample_count` |
| `latin_hypercube` | Seeded NumPy PCG64 permutation per variable |
| `sobol` | Seeded digital shift, 32-bit core, maximum 16 dimensions |

The same design-space contract, plan, seed, and sample index produce the same
candidate identity and order. Stochastic strategies change when the seed
changes. Invalid finite grids, duplicate decoded candidates, and Sobol requests
above 16 dimensions fail closed.

## Generation and resume

```python
factory = DatasetFactory(design_space, evaluator)
result = factory.generate(
    SamplingPlan(strategy="latin_hypercube", sample_count=32, seed=42),
    DatasetConfig(output_root="outputs/dataset", resume=True, cache=True),
)
```

Resume is strictly bound to the design-space ID, objective-set ID, sampling
plan, candidate count, candidate ordering, batch identity, and VeriTMM version.
Completed indexed rows are not rerun. A corrupt or mismatched manifest, index,
record, or artifact reference is rejected rather than repaired silently.

The batch executor is replaceable through the public `BatchExecutor` protocol.
The default sequential executor isolates candidate failures. A mixed outcome
produces `partial`; successful peers remain certificate-bound, while failed or
rejected rows have no certificate and cannot be upgraded by ML or optimizer
metadata.

## Artifact layout

```text
dataset/
  DATASET_MANIFEST.json       bounded identity and counts
  DATASET_INDEX.jsonl         one compact DatasetRecord per sample
  evaluations/
    BATCH_MANIFEST.json       batch binding, progress, counts
    BATCH_INDEX.jsonl         certificate-bound EvaluationRecords
    evaluations/
      e_<id>_<nonce>/         ordinary managed VeriTMM run artifacts
        RUN_RESULT.json
        RESULT_SUMMARY.json
        RESPONSE_CONTEXT.json
        SIMULATION_RESULT.json
        PHYSICS_ACCEPTANCE_CERTIFICATE.json
        ...
```

Paths in public results are relative POSIX paths with SHA-256 and byte size.
Validate them with `validate_artifact_references` before use.

## Compact records

`DatasetGenerationResult`, `DatasetManifest`, and `DatasetRecord` never embed
spectra, wavelength arrays, solver histories, sample populations, or
trajectories. A row keeps candidate/design values, normalized features,
material identities, compact wavelength configuration, requested/selected
outputs, task/run/catalog identities, verification status, certificate ID,
version, provenance, and artifact references.

Open the referenced `SIMULATION_RESULT.json` only when spectrum detail is
needed. “Full dataset” means all persisted compact rows and their valid
artifact links, not duplicated inline spectra.
