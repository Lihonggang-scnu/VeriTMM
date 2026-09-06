"""LegacyEvidenceAdapter: map published v1.0.0 artifacts onto the evidence model.

The adapter serves one purpose: let ``verify-run`` replay a historical
published-v1.0 run against the current evidence model.  Its contract is
deliberately narrow:

```text
recognized v1.0 artifact set
  -> map ONLY observation and audit fields that really exist on disk
  -> fields that do not exist stay unavailable (the mapping refuses)
  -> evaluate_evidence() re-derives the verdict
```

Two hard rules keep the replay honest:

1. **No verdict back-derivation.**  The adapter never reads ``accepted``,
   ``status``, ``failures``, ``evidence_coverage``, or ``certificate_id`` to
   construct evidence.  Reconstructed evidence comes from the observation
   artifacts (``NORMALIZED_TASK.json``, ``SIMULATION_RESULT.json``) and from
   the recorded audit/report fields of the certificate.  A tampered verdict is
   therefore caught by the replay comparison instead of being laundered into
   the evidence.
2. **Binding before replay.**  The mapped task payload must re-hash, under
   ``veritmm-canonical-json-v1``, to the task identity recorded in the legacy
   certificate, and the mapped raw material provenance must re-hash to the
   recorded ``material_provenance_sha256``.  Historical tasks whose recorded
   identity was produced under a different canonicalization (for example
   non-ASCII payloads hashed with the pre-1.1 ASCII-escaping convention) are
   reported as insufficient — they stay ``replay_status: unavailable`` rather
   than being silently re-hashed into agreement.

Insufficiency is a normal outcome: a run whose historical artifacts do not
carry everything the current evaluation consumes simply stays
``replay_status: unavailable`` — its historical certificate remains valid
history and is never rewritten.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .acceptance import EVIDENCE_SCHEMA_VERSION, VerificationEvidence
from .hashing import IDENTITY_SCHEME, stable_sha256


@dataclass(frozen=True)
class LegacyEvidenceMapping:
    """A successfully mapped legacy evidence object with its provenance."""

    evidence: VerificationEvidence
    mapped_fields: Tuple[str, ...]
    unavailable_fields: Tuple[str, ...]


def _load_artifact(
    root: Path,
    references_by_kind: Mapping[str, Mapping[str, Any]],
    kind: str,
) -> Optional[Dict[str, Any]]:
    reference = references_by_kind.get(kind)
    if reference is None:
        return None
    path = root / str(reference["path"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_legacy_evidence(
    root: str | Path,
    envelope: Mapping[str, Any],
    certificate: Mapping[str, Any],
    references_by_kind: Mapping[str, Mapping[str, Any]],
) -> Tuple[Optional[LegacyEvidenceMapping], List[str]]:
    """Map a published v1.0.0 artifact set onto ``VerificationEvidence``.

    Returns ``(None, reasons)`` when the historical artifacts do not carry
    everything the current evaluation consumes.  ``reasons`` explains each
    missing dimension; callers must surface them as
    ``replay_status: unavailable`` instead of approximating.
    """

    root_path = Path(root)
    reasons: List[str] = []

    normalized = _load_artifact(root_path, references_by_kind, "normalized_task")
    if normalized is None:
        reasons.append("NORMALIZED_TASK.json is missing or unreadable")
    simulation = _load_artifact(root_path, references_by_kind, "simulation_result")
    if simulation is None:
        reasons.append("SIMULATION_RESULT.json is missing or unreadable")
    if normalized is None or simulation is None:
        return None, reasons

    if str(normalized.get("mode")) != "simulate":
        reasons.append(
            f"the normalized task records mode {normalized.get('mode')!r}; "
            "only legacy simulate runs are supported by this adapter"
        )
    task_payload = normalized.get("simulation")
    if not isinstance(task_payload, dict):
        reasons.append("the normalized task does not contain a simulation payload")
        return None, reasons

    capability_assessment = certificate.get("capability_assessment")
    if not isinstance(capability_assessment, dict):
        reasons.append("the legacy certificate does not record a capability assessment")
        return None, reasons
    if capability_assessment.get("supported") is not True:
        reasons.append(
            "legacy capability rejections cannot be mapped without the "
            "historical raw failure records"
        )
        return None, reasons

    recorded_task_sha256 = certificate.get("task_sha256")
    recomputed_task_sha256 = stable_sha256(task_payload)
    if recorded_task_sha256 != recomputed_task_sha256:
        reasons.append(
            "the legacy task identity does not match a veritmm-canonical-json-v1 "
            "recomputation of the recorded task payload (expected for non-ASCII "
            "tasks hashed under the pre-1.1 convention)"
        )
        return None, reasons

    raw_provenance = simulation.get("material_provenance")
    recorded_provenance_sha256 = certificate.get("material_provenance_sha256")
    # The provenance is a per-layer record list in both eras; the container is
    # deliberately not constrained — the recorded digest is the binding.
    if raw_provenance is None or not isinstance(recorded_provenance_sha256, str):
        reasons.append(
            "the historical artifacts do not carry the raw material provenance "
            "recorded by the certificate"
        )
        return None, reasons
    if stable_sha256(raw_provenance) != recorded_provenance_sha256:
        reasons.append(
            "the recorded material provenance does not match the observation "
            "artifacts"
        )
        return None, reasons

    channels = simulation.get("channels")
    extras = simulation.get("extras")
    physics_audit = simulation.get("audit") or certificate.get("physics_audit")
    if not isinstance(channels, dict) or not isinstance(extras, dict):
        reasons.append("the simulation result does not record channels and extras")
        return None, reasons
    if not isinstance(physics_audit, dict):
        reasons.append("the historical artifacts do not record a physics audit")
        return None, reasons

    illumination = task_payload.get("illumination") or {}
    evidence = VerificationEvidence(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        task_payload=task_payload,
        task_sha256=recorded_task_sha256,
        veritmm_version=str(certificate.get("veritmm_version") or "1.0.0"),
        identity_scheme=IDENTITY_SCHEME,
        run_id=str(envelope.get("run_id") or ""),
        capability_assessment=capability_assessment,
        capability_failure_dicts=[],
        runtime=dict(certificate.get("runtime") or {}),
        requested_outputs=[str(item) for item in task_payload.get("requested_outputs", [])],
        illumination_angles_deg=[
            float(angle)
            for angle in (illumination.get("angles_deg") or [])
        ],
        channel_output_keys={
            str(key): sorted(str(name) for name in values.keys())
            for key, values in channels.items()
        },
        extras_keys=sorted(str(key) for key in extras.keys()),
        solver_name=(
            str(certificate.get("solver"))
            if certificate.get("solver") is not None
            else None
        ),
        physics_audit=physics_audit,
        material_provenance=raw_provenance,
        material_catalog=(
            dict(certificate["material_catalog"])
            if isinstance(certificate.get("material_catalog"), dict)
            else None
        ),
        spectral_convergence=(
            dict(certificate["spectral_convergence"])
            if isinstance(certificate.get("spectral_convergence"), dict)
            else None
        ),
        independent_solver_check=(
            dict(certificate["independent_solver_check"])
            if isinstance(certificate.get("independent_solver_check"), dict)
            else None
        ),
        tightest_margin=(
            dict(certificate["tightest_margin"])
            if isinstance(certificate.get("tightest_margin"), dict)
            else None
        ),
        high_precision_referee=(
            dict(certificate["high_precision_referee"])
            if isinstance(certificate.get("high_precision_referee"), dict)
            else None
        ),
        execution_error=None,
    )
    mapped = (
        "task_payload",
        "task_sha256",
        "capability_assessment",
        "runtime",
        "requested_outputs",
        "illumination_angles_deg",
        "channel_output_keys",
        "extras_keys",
        "solver_name",
        "physics_audit",
        "material_provenance",
        "material_catalog",
        "spectral_convergence",
        "independent_solver_check",
        "tightest_margin",
        "high_precision_referee",
    )
    unavailable = ("capability_failure_dicts", "execution_error")
    return (
        LegacyEvidenceMapping(
            evidence=evidence, mapped_fields=mapped, unavailable_fields=unavailable
        ),
        reasons,
    )


__all__ = ["LegacyEvidenceMapping", "build_legacy_evidence"]
