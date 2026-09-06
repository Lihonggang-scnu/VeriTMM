"""Unified physics backends: one KernelSpec, many execution engines."""

from .registry import (
    BackendNotRegisteredError,
    BackendResult,
    KernelSpec,
    available_backends,
    get_backend,
    register_backend,
    register_backend_search_path,
)

__all__ = [
    "BackendNotRegisteredError",
    "BackendResult",
    "KernelSpec",
    "available_backends",
    "get_backend",
    "register_backend",
    "register_backend_search_path",
]
