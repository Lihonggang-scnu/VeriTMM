"""Torch differentiable backend: a thin wrapper over ``DifferentiableTMM``.

Exposes the existing PyTorch S-matrix implementation (batched, autograd
capable) through the unified registry.  No physics lives here: the kernel is
adapted to the batched tensor shapes ``DifferentiableTMM`` already defines.
When PyTorch is absent, this module fails to import and the registry simply
reports the backend as not registered.  Absolute imports are required because
drop-in backend modules load under a synthetic module name.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from tmm_engine.backends.registry import BackendResult, KernelSpec
from tmm_engine.differentiable import DifferentiableTMM

BACKEND_NAME = "torch"

_DTYPES = {
    "float64": (torch.float64, torch.complex128),
    "float32": (torch.float32, torch.complex64),
}


def _as_tensor(value: Any, dtype: "torch.dtype") -> "torch.Tensor":
    """Tensor inputs pass through untouched (autograd graph preserved);
    numpy arrays and python sequences are converted with an explicit dtype —
    python-sequence conversion must never inherit the torch default
    (float32), which would quantize the phase thickness."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.as_tensor(np.ascontiguousarray(value), dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def assemble(kernel: KernelSpec, dtype: str = "float64") -> BackendResult:
    """Assemble one kernel through the differentiable torch S-matrix path.

    ``thicknesses_nm`` may carry ``requires_grad`` tensors — the autograd
    graph is preserved.  A 1-D (unbatched) kernel is promoted to a batch of
    one; batched kernels pass through unchanged.
    """

    if kernel.assembly != "smatrix":
        raise ValueError(f"unsupported assembly formula: {kernel.assembly!r}")
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported dtype: {dtype!r}")
    dtype_real, dtype_complex = _DTYPES[dtype]

    nk = kernel.nk_stack
    if not isinstance(nk, torch.Tensor):
        nk = np.stack([np.ascontiguousarray(np.asarray(row)) for row in nk])
    nk = _as_tensor(nk, dtype_complex)
    thicknesses = _as_tensor(kernel.thicknesses_nm, dtype_real)
    wavelengths = _as_tensor(kernel.wavelengths_nm, dtype_real)
    if nk.dim() == 2:
        nk = nk.unsqueeze(0)
    if thicknesses.dim() == 1:
        thicknesses = thicknesses.unsqueeze(0)
    model = DifferentiableTMM(
        polarization=kernel.polarization,
        dtype_real=dtype_real,
        dtype_complex=dtype_complex,
    )
    output = model(
        thicknesses * 1e-3,
        nk,
        wavelengths * 1e-3,
        math.radians(float(kernel.angle_deg)),
    )
    return BackendResult(R=output.R, T=output.T, A=output.A, r=output.r, t=output.t)
