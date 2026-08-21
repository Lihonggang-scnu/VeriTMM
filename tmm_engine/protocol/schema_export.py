"""JSON Schema export for the public VeriTMM protocol models."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel

from .models import (
    FailureRecordModel,
    OptimizationTaskContract,
    PreflightReport,
    ResponseMetadata,
    RunResultEnvelope,
    SensitivityTaskContract,
    SimulationTaskContract,
    SweepTaskContract,
    ToleranceTaskContract,
)

SchemaKind: TypeAlias = Literal[
    "simulation",
    "optimization",
    "sweep",
    "sensitivity",
    "tolerance",
    "preflight",
    "failure",
    "run_result",
    "response",
]

_SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "simulation": SimulationTaskContract,
    "optimization": OptimizationTaskContract,
    "sweep": SweepTaskContract,
    "sensitivity": SensitivityTaskContract,
    "tolerance": ToleranceTaskContract,
    "preflight": PreflightReport,
    "failure": FailureRecordModel,
    "run_result": RunResultEnvelope,
    "response": ResponseMetadata,
}


def export_schema(kind: SchemaKind | str) -> dict[str, object]:
    """Return a JSON Schema 2020-12 document for one public protocol kind.

    The returned mapping contains only JSON-serializable values and is safe to
    pass directly to :func:`json.dumps`.
    """

    if kind not in _SCHEMA_MODELS:
        supported = ", ".join(_SCHEMA_MODELS)
        raise ValueError(f"unknown schema kind {kind!r}; expected one of: {supported}")
    schema = _SCHEMA_MODELS[kind].model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


__all__ = ["SchemaKind", "export_schema"]
