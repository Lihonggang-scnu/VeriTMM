"""NumPy reference backend: a thin wrapper over the existing S-matrix solver.

This backend adds no physics.  It converts one KernelSpec into the exact
``tmm_engine.tmm_solver`` call path the workbench has always used, so the
reference numerical behaviour is untouched.  Drop-in backend modules load
under a synthetic module name during discovery, so all imports here are
absolute by necessity.
"""

from __future__ import annotations

import numpy as np

from tmm_engine.backends.registry import BackendResult, KernelSpec
from tmm_engine.tmm_solver import TMM, TMMConfig

BACKEND_NAME = "numpy"


def _assemble_one_pol(
    n_list: list,
    d_list_m: list,
    wavelengths_um,
    angle_deg: float,
    polarization: str,
    n_incident,
):
    solver = TMM(
        TMMConfig(
            wavelength_unit="um",
            n_incident=n_incident,
            treat_last_layer_as_substrate=True,
            clip_R=False,
            clip_T=False,
        )
    )
    return solver.rt_spectrum(
        n_list,
        d_list_m,
        wavelengths_um,
        float(angle_deg),
        polarization,
        "deg",
        return_amplitudes=True,
    )


def assemble(kernel: KernelSpec) -> BackendResult:
    """Assemble one kernel through the reference NumPy S-matrix path."""

    if kernel.assembly != "smatrix":
        raise ValueError(f"unsupported assembly formula: {kernel.assembly!r}")
    if kernel.polarization not in ("s", "p", "unpolarized"):
        raise ValueError(f"unsupported polarization: {kernel.polarization!r}")

    n_incident = np.asarray(kernel.nk_stack[0], dtype=np.complex128)
    n_list = [np.asarray(nk, dtype=np.complex128) for nk in kernel.nk_stack[1:]]
    d_list_m = [float(thickness) * 1e-9 for thickness in kernel.thicknesses_nm]
    d_list_m = d_list_m + [0.0]
    wavelengths_um = np.asarray(kernel.wavelengths_nm, dtype=np.float64) * 1e-3

    if kernel.polarization == "unpolarized":
        rs, ts, r_s, t_s = _assemble_one_pol(
            n_list, d_list_m, wavelengths_um, kernel.angle_deg, "s", n_incident
        )
        rp, tp, r_p, t_p = _assemble_one_pol(
            n_list, d_list_m, wavelengths_um, kernel.angle_deg, "p", n_incident
        )
        R = 0.5 * (np.asarray(rs) + np.asarray(rp))
        T = 0.5 * (np.asarray(ts) + np.asarray(tp))
        r = np.stack([np.asarray(r_s), np.asarray(r_p)], axis=0)
        t = np.stack([np.asarray(t_s), np.asarray(t_p)], axis=0)
    else:
        R, T, r, t = _assemble_one_pol(
            n_list,
            d_list_m,
            wavelengths_um,
            kernel.angle_deg,
            kernel.polarization,
            n_incident,
        )
        R, T = np.asarray(R), np.asarray(T)
        r, t = np.asarray(r), np.asarray(t)
    A = 1.0 - R - T
    return BackendResult(R=R, T=T, A=A, r=r, t=t)
