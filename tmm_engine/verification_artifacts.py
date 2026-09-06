"""Persisted verification evidence and policy artifacts.

Three artifacts carry one acceptance decision:

```text
VERIFICATION_EVIDENCE.json  — observed facts (raw decision inputs)
VERIFICATION_POLICY.json    — judgement rules (tolerances, audit switches)
certificate                 — the verdict recomputed from the two
```

``VERIFICATION_EVIDENCE.json`` binds ``schema_version``, ``identity_scheme``,
``run_id``, and ``task_sha256``; loading re-validates all four so evidence from
one run can never be spliced onto another run's certificate.  Neither artifact
references ``RUN_RESULT`` or any other artifact, so their hashes stay
self-contained (the same pattern as ``RESPONSE_CONTEXT.json``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from .acceptance import (
    EVIDENCE_SCHEMA_VERSION,
    AcceptanceSettings,
    VerificationArtifactError,
    VerificationEvidence,
    verification_policy_dict,
    verification_policy_from_dict,
)
from .hashing import IDENTITY_SCHEME
from .run_artifacts import write_json

VERIFICATION_EVIDENCE_FILENAME = "VERIFICATION_EVIDENCE.json"
VERIFICATION_POLICY_FILENAME = "VERIFICATION_POLICY.json"

_EVIDENCE_FIELD_NAMES = tuple(
    item.name for item in dataclasses.fields(VerificationEvidence)
)

# Fields added after the evidence schema was frozen.  Their absence marks
# evidence written by an earlier release, which is loaded as ``None`` and
# judged under the pre-ledger (closure-only) semantics instead of being
# rejected: old run directories must keep replaying.
_EVIDENCE_FIELDS_OPTIONAL_ON_LOAD = frozenset({"energy_accounting", "reciprocity_check"})


def write_verification_artifacts(
    output_dir: str | Path,
    evidence: VerificationEvidence,
    settings: AcceptanceSettings,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write the evidence and policy pair for one certified run.

    The run identity is stamped into the evidence at write time; the facts
    themselves are untouched.  Both files are written before ``RUN_RESULT``
    is indexed, so their artifact references carry stable hashes and neither
    file can reference the index that hashes it.
    """

    if not isinstance(run_id, str) or not run_id:
        raise VerificationArtifactError("run_id must be a non-empty string")
    stamped = dataclasses.replace(evidence, run_id=run_id)
    evidence_payload = stamped.to_dict()
    policy_payload = verification_policy_dict(settings)
    root = Path(output_dir)
    write_json(root / VERIFICATION_EVIDENCE_FILENAME, evidence_payload)
    write_json(root / VERIFICATION_POLICY_FILENAME, policy_payload)
    return evidence_payload, policy_payload


def _read_json_object(path: str | Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationArtifactError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise VerificationArtifactError(f"{label} must contain a JSON object")
    return payload


def load_verification_evidence(
    path: str | Path,
    *,
    run_id: Optional[str] = None,
    task_sha256: Optional[str] = None,
) -> VerificationEvidence:
    """Load evidence and re-validate its identity binding.

    Fails closed on an unsupported ``schema_version``, a foreign
    ``identity_scheme``, unknown or missing fields, and — when the caller
    supplies expected values — a ``run_id`` or ``task_sha256`` mismatch.
    """

    payload = _read_json_object(path, "verification evidence")
    if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise VerificationArtifactError(
            "verification evidence has an unsupported schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("identity_scheme") != IDENTITY_SCHEME:
        raise VerificationArtifactError(
            "verification evidence identity_scheme mismatch: "
            f"{payload.get('identity_scheme')!r}"
        )
    evidence_run_id = payload.get("run_id")
    if not isinstance(evidence_run_id, str) or not evidence_run_id:
        raise VerificationArtifactError(
            "verification evidence is missing its run_id binding"
        )
    if run_id is not None and evidence_run_id != str(run_id):
        raise VerificationArtifactError(
            f"verification evidence is bound to run {evidence_run_id!r}, "
            f"not {str(run_id)!r}"
        )
    evidence_task_sha256 = payload.get("task_sha256")
    if (
        not isinstance(evidence_task_sha256, str)
        or len(evidence_task_sha256) != 64
    ):
        raise VerificationArtifactError(
            "verification evidence is missing a valid task_sha256 binding"
        )
    if task_sha256 is not None and evidence_task_sha256 != str(task_sha256):
        raise VerificationArtifactError(
            "verification evidence is bound to a different simulation task"
        )
    unknown = sorted(set(payload) - set(_EVIDENCE_FIELD_NAMES))
    if unknown:
        raise VerificationArtifactError(
            f"verification evidence has unknown fields: {unknown}"
        )
    missing = sorted(
        set(_EVIDENCE_FIELD_NAMES) - set(payload) - _EVIDENCE_FIELDS_OPTIONAL_ON_LOAD
    )
    if missing:
        raise VerificationArtifactError(
            f"verification evidence is missing fields: {missing}"
        )
    for optional_object_field in ("energy_accounting", "reciprocity_check"):
        if optional_object_field in payload:
            value = payload.get(optional_object_field)
            if value is not None and not isinstance(value, dict):
                raise VerificationArtifactError(
                    f"verification evidence field {optional_object_field!r} must "
                    "be a JSON object when present"
                )
    return VerificationEvidence(**payload)


def load_verification_policy(path: str | Path) -> AcceptanceSettings:
    """Load a persisted verification policy back into acceptance settings."""

    payload = _read_json_object(path, "verification policy")
    return verification_policy_from_dict(payload)


__all__ = [
    "VERIFICATION_EVIDENCE_FILENAME",
    "VERIFICATION_POLICY_FILENAME",
    "VerificationArtifactError",
    "load_verification_evidence",
    "load_verification_policy",
    "write_verification_artifacts",
]
