"""Offline four-state verification of one persisted run directory.

``verify-run`` closes the trust loop for a third party who holds only the run
directory: it re-checks the artifact chain, re-derives the verdict from the
persisted evidence and policy, and reports what it found as four independent
states plus one convenience summary:

```text
integrity_status       valid | invalid          files self-consistent with the index
certification_status   certified | uncertified | not_evaluated
                       did this run obtain a full physics acceptance certificate
replay_status          passed | failed | unavailable | not_evaluated
                       can evaluate_evidence() reproduce the verdict here
authenticity_status    unsigned | valid | invalid   who signed the artifact set
overall_status         valid | invalid | incomplete   machine-consumable summary
```

The states are deliberately independent.  A cache replay carries a valid
certificate issued by an earlier run but has no evidence of its own, so it is
``certified`` while ``replay_status`` is ``unavailable``.  ``uncertified`` is
reserved for results that never obtained a full acceptance certificate (for
example a ``skip_certificate`` run, detected by both acceptance checks being
absent from the certificate itself).

Evaluation order is fixed.  Once artifact integrity fails, the content of
evidence, policy, and certificate is no longer trusted: the report stops with
integrity diagnostics and marks the remaining states ``not_evaluated`` rather
than producing misleading secondary judgements from tampered material.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .acceptance import VerificationArtifactError, evaluate_evidence
from .hashing import IDENTITY_SCHEME, canonical_json_bytes, file_sha256
from .verification_artifacts import (
    load_verification_evidence,
    load_verification_policy,
)

VERIFY_RUN_REPORT_SCHEMA_VERSION = "veritmm-verify-run-report-v1"

RUN_RESULT_SCHEMA = "veritmm-run-result-v1"

_REQUIRED_REFERENCE_FIELDS = ("kind", "path", "schema_version", "sha256", "size_bytes")


def _fail(
    report: Dict[str, Any],
    code: str,
    message: str,
) -> Dict[str, Any]:
    report["integrity_status"] = "invalid"
    report["integrity_failures"].append({"code": code, "message": message})
    report["certification_status"] = "not_evaluated"
    report["replay_status"] = "not_evaluated"
    report["overall_status"] = "invalid"
    return report


def _reference_issues(
    root: Path,
    reference: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Structural, size, and digest checks for one artifact reference."""

    issues: List[Dict[str, Any]] = []
    label = str(reference.get("kind") or reference.get("path") or "<reference>")
    for field in _REQUIRED_REFERENCE_FIELDS:
        if field not in reference:
            issues.append(
                {"code": "reference_field_missing", "artifact": label,
                 "message": f"artifact reference is missing {field!r}"}
            )
    if issues:
        return issues
    raw_path = str(reference["path"])
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        issues.append(
            {"code": "reference_path_escapes_root", "artifact": label,
             "message": f"artifact path escapes the run directory: {raw_path}"}
        )
        return issues
    if not candidate.is_file():
        issues.append(
            {"code": "artifact_missing", "artifact": label,
             "message": f"indexed artifact is not a file: {raw_path}"}
        )
        return issues
    actual_size = candidate.stat().st_size
    if int(reference["size_bytes"]) != actual_size:
        issues.append(
            {"code": "size_mismatch", "artifact": label,
             "message": (
                 f"indexed size {reference['size_bytes']} does not match "
                 f"on-disk size {actual_size}: {raw_path}"
             )}
        )
    actual_sha256 = file_sha256(candidate)
    if str(reference["sha256"]) != actual_sha256:
        issues.append(
            {"code": "sha256_mismatch", "artifact": label,
             "message": f"indexed sha256 does not match file content: {raw_path}"}
        )
    return issues


def _verdict_core_differences(
    replayed: Mapping[str, Any],
    persisted: Mapping[str, Any],
) -> List[str]:
    """Fields of the verdict that must reproduce across schema versions."""

    differences: List[str] = []
    for key in (
        "accepted",
        "status",
        "task_sha256",
        "solver",
        "physics_audit",
        "spectral_convergence",
        "independent_solver_check",
    ):
        if replayed.get(key) != persisted.get(key):
            differences.append(key)
    replayed_codes = [
        item.get("code") for item in (replayed.get("failures") or []) if isinstance(item, dict)
    ]
    persisted_codes = [
        item.get("code") for item in (persisted.get("failures") or []) if isinstance(item, dict)
    ]
    if replayed_codes != persisted_codes:
        differences.append("failures")
    for optional in ("tightest_margin", "high_precision_referee"):
        if optional in persisted and replayed.get(optional) != persisted.get(optional):
            differences.append(optional)
    return differences


def verify_run_dir(
    run_dir: str | Path,
    *,
    expected_run_id: Optional[str] = None,
    expected_task_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify one run directory and return the four-state report."""

    root = Path(run_dir)
    report: Dict[str, Any] = {
        "schema_version": VERIFY_RUN_REPORT_SCHEMA_VERSION,
        "identity_scheme": IDENTITY_SCHEME,
        "run_dir": str(root),
        "run_id": None,
        "task_sha256": None,
        "certificate_task_sha256": None,
        "certificate_id": None,
        "integrity_status": "valid",
        "integrity_failures": [],
        "certification_status": "not_evaluated",
        "certification_reasons": [],
        "certificate_verdict": None,
        "acceptance_checks": None,
        "replay_status": "unavailable",
        "replay_reason": None,
        "replay_differences": [],
        "replay_comparison_scope": None,
        "legacy_replay": None,
        "authenticity_status": "unsigned",
        "authenticity_reason": (
            "no signature verification is available in this release; the "
            "DSSE attestation layer arrives with the 1.2 signing work"
        ),
        "overall_status": "invalid",
    }

    # 1. Parse the run envelope.
    envelope_path = root / "RUN_RESULT.json"
    if not envelope_path.is_file():
        return _fail(report, "run_envelope_missing",
                     "the run directory does not contain RUN_RESULT.json")
    try:
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(report, "run_envelope_unreadable",
                     f"RUN_RESULT.json is unreadable: {exc}")
    if not isinstance(envelope, dict):
        return _fail(report, "run_envelope_malformed",
                     "RUN_RESULT.json must contain a JSON object")
    if envelope.get("schema_version") != RUN_RESULT_SCHEMA:
        report["integrity_failures"].append(
            {"code": "unsupported_envelope_schema",
             "message": (
                 f"unsupported run envelope schema_version: "
                 f"{envelope.get('schema_version')!r}"
             )}
        )
    run_id = envelope.get("run_id")
    task_sha256 = envelope.get("task_sha256")
    envelope_certificate_id = envelope.get("certificate_id")
    if not isinstance(run_id, str) or not run_id:
        return _fail(report, "run_identity_missing",
                     "the run envelope does not carry a run_id")
    if not isinstance(task_sha256, str) or len(task_sha256) != 64:
        return _fail(report, "task_identity_missing",
                     "the run envelope does not carry a task_sha256")
    report["run_id"] = run_id
    report["task_sha256"] = task_sha256
    report["certificate_id"] = (
        envelope_certificate_id if isinstance(envelope_certificate_id, str) else None
    )
    if expected_run_id is not None and run_id != str(expected_run_id):
        return _fail(report, "run_identity_mismatch",
                     f"the directory holds run {run_id!r}, expected "
                     f"{str(expected_run_id)!r}")
    if expected_task_sha256 is not None and task_sha256 != str(expected_task_sha256):
        return _fail(report, "task_identity_mismatch",
                     "the directory holds a different simulation task than expected")

    # 2+3. Structure, size, and digest integrity of every indexed artifact.
    references = envelope.get("artifacts")
    if not isinstance(references, list) or not references:
        return _fail(report, "artifact_index_missing",
                     "the run envelope does not carry a non-empty artifact index")
    for reference in references:
        if not isinstance(reference, dict):
            report["integrity_failures"].append(
                {"code": "reference_malformed", "artifact": "<non-object>",
                 "message": "artifact index entries must be JSON objects"}
            )
            continue
        report["integrity_failures"].extend(_reference_issues(root, reference))
    if report["integrity_failures"]:
        # Stop here: nothing downstream may be trusted after an integrity
        # failure, so no scientific re-judgement is attempted.
        return _fail(report, "integrity_failed",
                     "artifact integrity failures prevent content evaluation")

    references_by_kind = {str(ref["kind"]): ref for ref in references}
    legacy_run = "identity_scheme" not in envelope

    # 4. Certification state, from the certificate artifact itself.
    persisted_certificate: Optional[Dict[str, Any]] = None
    certificate_reference = references_by_kind.get("physics_certificate")
    if certificate_reference is None:
        report["certification_status"] = "uncertified"
        report["certification_reasons"].append(
            "the run does not contain a physics acceptance certificate artifact"
        )
    else:
        certificate_path = root / str(certificate_reference["path"])
        try:
            loaded = json.loads(certificate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _fail(report, "certificate_unreadable",
                         f"the certificate artifact is unreadable: {exc}")
        if not isinstance(loaded, dict):
            return _fail(report, "certificate_malformed",
                         "the certificate artifact must contain a JSON object")
        # The certificate carries the raw-task identity while the envelope
        # carries the normalized-operation identity; the two scopes meet
        # through the evidence (checked below) and the certificate_id.
        certificate_task_sha256 = loaded.get("task_sha256")
        if not isinstance(certificate_task_sha256, str) or len(certificate_task_sha256) != 64:
            return _fail(report, "certificate_task_identity_missing",
                         "the certificate does not carry a task_sha256")
        report["certificate_task_sha256"] = certificate_task_sha256
        if envelope_certificate_id is not None and (
            loaded.get("certificate_id") != envelope_certificate_id
        ):
            return _fail(report, "certificate_identity_mismatch",
                         "the run envelope and the certificate disagree on "
                         "certificate_id")
        persisted_certificate = loaded
        report["certificate_id"] = loaded.get("certificate_id")
        spectral = loaded.get("spectral_convergence") or {}
        independent = loaded.get("independent_solver_check") or {}
        spectral_ran = spectral.get("status") not in (None, "not_requested")
        independent_ran = independent.get("status") not in (None, "not_requested")
        report["acceptance_checks"] = {
            "spectral_convergence": spectral.get("status"),
            "independent_solver": independent.get("status"),
        }
        if loaded.get("accepted") is True:
            report["certificate_verdict"] = "accepted"
        elif loaded.get("accepted") is False:
            report["certificate_verdict"] = "rejected"
        if spectral_ran or independent_ran:
            report["certification_status"] = "certified"
        else:
            report["certification_status"] = "uncertified"
            report["certification_reasons"].append(
                "the certificate was issued without full acceptance checks: "
                "spectral convergence and the independent solver were both "
                "not requested"
            )

    # 5-7. Replay from the persisted evidence and policy.
    evidence_reference = references_by_kind.get("verification_evidence")
    policy_reference = references_by_kind.get("verification_policy")
    if legacy_run:
        if persisted_certificate is None:
            report["replay_status"] = "unavailable"
            report["replay_reason"] = (
                "legacy run without a physics certificate artifact; there is "
                "no verdict to replay"
            )
        else:
            from .legacy_evidence import build_legacy_evidence

            mapping, legacy_reasons = build_legacy_evidence(
                root, envelope, persisted_certificate, references_by_kind
            )
            if mapping is None:
                report["replay_status"] = "unavailable"
                report["replay_reason"] = (
                    "the legacy artifacts provide insufficient evidence for a "
                    "replay: " + "; ".join(legacy_reasons)
                )
                report["legacy_replay"] = {"reasons": legacy_reasons}
            else:
                # The legacy acceptance policy is not persisted; the default
                # settings are applied and a verdict-core comparison tolerates
                # schema evolution while still failing on any scientific
                # difference.
                replayed = evaluate_evidence(mapping.evidence)
                if canonical_json_bytes(replayed) == canonical_json_bytes(
                    persisted_certificate
                ):
                    report["replay_status"] = "passed"
                    comparison_scope = "full"
                else:
                    core_differences = _verdict_core_differences(
                        replayed, persisted_certificate
                    )
                    if not core_differences:
                        report["replay_status"] = "passed"
                        comparison_scope = "verdict_core"
                    else:
                        report["replay_status"] = "failed"
                        comparison_scope = "verdict_core"
                        report["replay_reason"] = (
                            "the replayed legacy verdict does not match the "
                            "persisted certificate; note that the legacy "
                            "acceptance policy is not persisted and default "
                            "settings were applied"
                        )
                        report["replay_differences"] = core_differences
                report["legacy_replay"] = {
                    "mapped_fields": list(mapping.mapped_fields),
                    "unavailable_fields": list(mapping.unavailable_fields),
                    "comparison_scope": comparison_scope,
                }
    elif evidence_reference is None or policy_reference is None:
        report["replay_status"] = "unavailable"
        if envelope.get("cache_hit") is True or envelope.get("source_run_id"):
            report["replay_reason"] = (
                "this run is a cache replay carrying a certificate issued by "
                f"an earlier run (source_run_id={envelope.get('source_run_id')!r}) "
                "and has no verification evidence of its own"
            )
        else:
            report["replay_reason"] = (
                "the run does not persist verification evidence and policy"
            )
    elif persisted_certificate is None:
        report["replay_status"] = "unavailable"
        report["replay_reason"] = (
            "evidence is present but there is no persisted certificate to "
            "compare the replayed verdict against"
        )
    else:
        try:
            evidence = load_verification_evidence(
                root / str(evidence_reference["path"]),
                run_id=run_id,
            )
            policy = load_verification_policy(
                root / str(policy_reference["path"])
            )
        except VerificationArtifactError as exc:
            return _fail(report, "verification_artifact_invalid",
                         f"the persisted verification artifacts are not usable: {exc}")
        if evidence.task_sha256 != persisted_certificate.get("task_sha256"):
            return _fail(report, "evidence_certificate_task_mismatch",
                         "the persisted evidence is bound to a different "
                         "simulation task than the certificate")
        try:
            replayed = evaluate_evidence(evidence, policy)
        except Exception as exc:  # noqa: BLE001 - judgement must never raise
            report["replay_status"] = "failed"
            report["replay_reason"] = (
                f"evaluate_evidence failed on the persisted evidence: {exc}"
            )
        else:
            if (
                canonical_json_bytes(replayed)
                == canonical_json_bytes(persisted_certificate)
                and replayed.get("certificate_id")
                == persisted_certificate.get("certificate_id")
            ):
                report["replay_status"] = "passed"
                report["replay_comparison_scope"] = "full"
            elif evidence.energy_accounting is None:
                # Evidence written before the independent energy ledger existed:
                # a newer engine replays it with the additive certificate block,
                # so the comparison tolerates additive schema evolution on the
                # same verdict core used for legacy runs.  Any scientific
                # difference still fails.
                core_differences = _verdict_core_differences(
                    replayed, persisted_certificate
                )
                if not core_differences:
                    report["replay_status"] = "passed"
                    report["replay_comparison_scope"] = "verdict_core"
                else:
                    report["replay_status"] = "failed"
                    report["replay_reason"] = (
                        "the replayed certificate does not match the persisted "
                        "certificate"
                    )
                    report["replay_differences"] = core_differences
            else:
                report["replay_status"] = "failed"
                report["replay_reason"] = (
                    "the replayed certificate does not match the persisted "
                    "certificate"
                )
                report["replay_differences"] = sorted(
                    {
                        *{
                            key
                            for key in set(replayed) | set(persisted_certificate)
                            if replayed.get(key) != persisted_certificate.get(key)
                        }
                    }
                )

    # 8. Authenticity stays "unsigned" until the 1.2 DSSE layer lands.

    if report["integrity_status"] == "invalid" or report["replay_status"] == "failed":
        report["overall_status"] = "invalid"
    elif (
        report["certification_status"] == "certified"
        and report["replay_status"] == "passed"
    ):
        report["overall_status"] = "valid"
    else:
        report["overall_status"] = "incomplete"
    return report


__all__ = [
    "VERIFY_RUN_REPORT_SCHEMA_VERSION",
    "verify_run_dir",
]
