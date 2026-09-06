"""Unified physics backend registry.

The physics of one coherent stack assembly is described by a
:class:`KernelSpec` — a plain, array-library-agnostic container (this module
imports neither numpy nor torch).  Backends consume the same description and
return :class:`BackendResult`; the numpy reference backend wraps the existing
``tmm_engine.tmm_solver`` S-matrix path unchanged, and the torch backend wraps
``tmm_engine.differentiable.DifferentiableTMM`` (batched, autograd-capable).

Backends are discovered lazily: every sibling module named ``*_backend.py``
that declares ``BACKEND_NAME`` is a drop-in backend.  Adding a backend (for
example jax) therefore means adding one file — no existing code changes.
A backend that is absent or not importable is reported as
:class:`BackendNotRegisteredError` by :func:`get_backend`, never as a raw
``ImportError``.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

__all__ = [
    "BackendNotRegisteredError",
    "BackendResult",
    "KernelSpec",
    "available_backends",
    "get_backend",
    "register_backend",
    "register_backend_search_path",
]

# Re-run discovery when a new search path is registered or a drop-in file is
# expected; discovered names are cached until then.
_discovered: Dict[str, Callable[[], Any]] = {}
_extra_search_paths: List[Path] = []


class BackendNotRegisteredError(KeyError):
    """Raised when a backend name is not registered or not importable.

    Subclasses :class:`KeyError` so callers can treat an unavailable backend
    as a lookup miss; a missing optional dependency therefore surfaces as a
    typed not-registered condition instead of a raw ``ImportError``.
    """


@dataclass(frozen=True)
class KernelSpec:
    """Data-driven description of one coherent stack assembly.

    A kernel holds everything an ``assemble`` implementation needs — the
    complex refractive index of every medium (ordered incident, films…, exit),
    the finite-film thicknesses, the wavelength grid, the incidence angle, and
    the polarization — plus the name of the assembly formula family.  The
    container is deliberately array-library-agnostic: fields may hold plain
    numbers, numpy arrays, or torch tensors, and the consuming backend
    interprets them natively (so torch tensors keep their autograd graph).
    Lengths use the project convention (nanometres); shape conventions are
    per-backend (the numpy reference backend is unbatched, the torch backend
    accepts and returns a leading batch dimension).
    """

    nk_stack: Any
    thicknesses_nm: Any
    wavelengths_nm: Any
    angle_deg: float = 0.0
    polarization: str = "unpolarized"
    assembly: str = "smatrix"


@dataclass
class BackendResult:
    """R/T/A (and optional r/t amplitudes) produced by one backend assembly."""

    R: Any
    T: Any
    A: Any
    r: Optional[Any] = None
    t: Optional[Any] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def _module_loader(module: Union[str, Any]) -> Callable[[], Any]:
    if isinstance(module, str):
        return lambda: importlib.import_module(module)
    if callable(module):
        return module
    return lambda: module


def register_backend(name: str, module: Union[str, Any]) -> None:
    """Register (or replace) a backend under ``name``.

    ``module`` may be an import path, a module object, or a zero-argument
    loader.  Registration is idempotent for the same target; the registry is
    the only place a backend is looked up, so an optional dependency that
    fails to import is reported as not-registered, never as an import error.
    """

    if not name:
        raise ValueError("backend name must be non-empty")
    _discovered[name] = _module_loader(module)


def register_backend_search_path(path: Union[str, Path]) -> None:
    """Add a directory scanned for ``*_backend.py`` drop-in modules."""

    candidate = Path(path)
    if candidate not in _extra_search_paths:
        _extra_search_paths.append(candidate)
    _discovered.clear()
    _discover()


def _iter_backend_modules():
    package_dir = Path(__file__).resolve().parent
    seen = set()
    for root in [package_dir, *_extra_search_paths]:
        if not root.is_dir():
            continue
        for info in pkgutil.iter_modules([str(root)]):
            if not info.name.endswith("_backend") or info.name in seen:
                continue
            seen.add(info.name)
            yield importlib.util.spec_from_file_location(
                f"_veritmm_backend_{info.name}", root / f"{info.name}.py"
            )


def _discover() -> None:
    for spec in _iter_backend_modules():
        if spec is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            # An optional dependency that is absent (e.g. torch not
            # installed) or a broken third-party drop-in is simply not
            # available; it must never surface as a raw import error.
            continue
        name = getattr(module, "BACKEND_NAME", None)
        if isinstance(name, str) and name:
            _discovered[name] = lambda module=module: module


def _ensure_discovered() -> None:
    if not _discovered:
        _discover()


def get_backend(name: str):
    """Return the backend module registered under ``name``."""

    _ensure_discovered()
    try:
        loader = _discovered[name]
    except KeyError:
        raise BackendNotRegisteredError(
            f"backend {name!r} is not registered; available backends: "
            f"{available_backends()}"
        ) from None
    return loader()


def available_backends() -> tuple:
    """Sorted names of the currently registered (and importable) backends."""

    _ensure_discovered()
    return tuple(sorted(_discovered))
