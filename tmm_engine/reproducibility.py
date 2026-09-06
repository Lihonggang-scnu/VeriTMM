"""Reproducibility semantics: two declared levels and an environment fingerprint.

VeriTMM separates two honest reproducibility claims (frozen 1.1 plan):

- **Identity level** (``byte_identical_within_scheme``): normalized tasks,
  schema documents, canonical metadata, and every canonical-JSON-derived hash
  are byte-identical across platforms *under the declared
  ``identity_scheme``* — guaranteed by construction of the canonical
  serializer.  The guarantee is scoped to the same identity scheme and
  protocol semantics; it does not survive a canonicalization change (which
  would bump the scheme).
- **Numerical level** (``tolerance_equivalent``): R/T/A, fields, optimization
  results, and verifier inputs agree across platforms within the declared
  acceptance tolerances (the same tolerance class the cross-solver gate
  uses).  Bit-identical solver output is *not* claimed without full
  environment pinning (Python/NumPy/BLAS versions plus CPU architecture),
  which is a separate, future contract.

The environment fingerprint records what produced a run (Python, NumPy,
best-effort BLAS description, platform, byte order) so a consumer can judge
which claims apply to it.
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict

from .hashing import IDENTITY_SCHEME

IDENTITY_LEVEL = "byte_identical_within_scheme"
NUMERICAL_LEVEL = "tolerance_equivalent"


def _blas_description() -> str:
    """Best-effort BLAS description; absence is recorded, never guessed."""

    try:
        import numpy

        try:
            config = numpy.show_config(mode="dicts") or {}
            blas = (config.get("Build Dependencies") or {}).get("blas") or {}
            name = blas.get("name")
            version = blas.get("version")
            if name:
                return f"{name} {version or '?'}".strip()
        except (TypeError, KeyError, AttributeError):
            pass
        try:
            from numpy.__config__ import get_info

            for key in ("blas_ilp64_opt", "blas_opt", "blas"):
                info = get_info(key) or {}
                parts = [str(info[key]) for key in ("name", "version") if info.get(key)]
                if parts:
                    return " ".join(parts)
        except (ImportError, KeyError):
            pass
        return "unknown"
    except Exception:  # pragma: no cover - numpy import failure is not a run stopper
        return "unknown"


def environment_fingerprint() -> Dict[str, str]:
    """Describe the environment that produced (or is inspecting) a run."""

    try:
        import numpy

        numpy_version = numpy.__version__
    except Exception:  # pragma: no cover
        numpy_version = "unknown"
    return {
        "python": sys.version.split()[0],
        "numpy": numpy_version,
        "blas": _blas_description(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
    }


def reproducibility_block() -> Dict[str, Any]:
    """The run-envelope traceability block describing reproducibility claims.

    Deliberately lean: the scope explanations live in the module docstring
    and ``docs/VALIDATION.md`` so the envelope stays inside the compact
    response budget.
    """

    return {
        "identity_scheme": IDENTITY_SCHEME,
        "identity_level": IDENTITY_LEVEL,
        "numerical_level": NUMERICAL_LEVEL,
        "environment": environment_fingerprint(),
    }


__all__ = [
    "IDENTITY_LEVEL",
    "NUMERICAL_LEVEL",
    "environment_fingerprint",
    "reproducibility_block",
]
