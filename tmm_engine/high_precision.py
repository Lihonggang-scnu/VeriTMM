"""High-precision TMM referee using mpmath (optional dependency).

When two float64 solvers disagree, or when tightest_margin shows a result
barely passed its acceptance threshold, this module recomputes R and T using
the characteristic-matrix method at quad-precision arithmetic (113 bits,
≈ 34 significant decimal digits).

The referee result is PURELY INFORMATIONAL — it never overrides or relaxes the
physics-acceptance decision made by the certificate.  Its purpose is to
identify which of the two float64 solvers (smatrix / byrnes) is closer to the
high-precision referee reference.

Conventions match tmm_solver.TMM.rt_spectrum_cm exactly:
  - eta_s = n * cos(theta)   for s-polarisation
  - eta_p = n / cos(theta)   for p-polarisation
  - Layer matrix M_i:
        [[cos(δ),    i·sin(δ)/η],
         [i·η·sin(δ), cos(δ)   ]]
    where δ = 2π n_i d_i cos(θ_i) / λ
  - r = (η0·B - C) / (η0·B + C),  t = 2η0 / (η0·B + C)
    with [B, C]^T = M · [1, η_s]^T
  - T = Re(η_s) / Re(η0) · |t|²

Requirements
------------
  pip install mpmath>=1.3   (or: pip install veritmm[high_precision])
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

_MPMATH_AVAILABLE = False
try:
    import mpmath as _mp  # type: ignore[import-unresolved]
    _MPMATH_AVAILABLE = True
except ImportError:
    pass

# Quad-precision: 113 mantissa bits ≈ 34 significant decimal digits
PRECISION_BITS: int = 113
# Trigger when normalized_margin < this (result barely passed)
TRIGGER_THRESHOLD: float = 0.20


def is_available() -> bool:
    """Return True if mpmath is installed and the referee can be used."""
    return _MPMATH_AVAILABLE


def _cos_layer(n_i: Any, n0: Any, sin_theta0: Any, mp: Any) -> Any:
    """Branch-cut-safe cos(theta_i) for layer i.

    Matches the convention in tmm_solver.TMM._cos_theta:
    choose the sign so that Im(n_i * cos_theta_i) >= 0  (forward-propagating).
    """
    sin_t = n0 * sin_theta0 / n_i
    cos_t = mp.sqrt(1 - sin_t * sin_t)
    kz = n_i * cos_t
    if mp.im(kz) < 0:
        cos_t = -cos_t
    return cos_t


def compute_rt_single_channel(
    *,
    n_all: Optional[Sequence[complex]] = None,
    n_by_wavelength: Optional[Sequence[Sequence[complex]]] = None,
    d_nm: Sequence[float],
    wavelengths_nm: Sequence[float],
    angle_deg: float,
    polarization: str,
    precision_bits: int = PRECISION_BITS,
) -> Dict[str, Any]:
    """Compute R(λ) and T(λ) at high precision for one angle/polarisation channel.

    Parameters
    ----------
    n_all : sequence of complex, optional
        Scalar (wavelength-independent) refractive indices for
        [incident_medium, layer_1, ..., layer_N, exit_medium].
        Length must equal len(d_nm) + 2.  Use for non-dispersive stacks.
        Exactly one of *n_all* or *n_by_wavelength* must be provided.
    n_by_wavelength : sequence of sequences of complex, optional
        Per-wavelength refractive indices.  ``n_by_wavelength[i][j]`` is the
        complex refractive index of medium *i* at wavelength index *j*.
        Length must equal len(d_nm) + 2; each inner sequence must have length
        len(wavelengths_nm).  Preferred over *n_all* for dispersive materials.
    d_nm : sequence of float
        Layer thicknesses in nm (internal layers only; media are semi-infinite).
    wavelengths_nm : sequence of float
        Wavelengths in nm.
    angle_deg : float
        Angle of incidence in degrees (0 ≤ angle < 90).
    polarization : str
        's' or 'p'.
    precision_bits : int
        mpmath working precision in bits (default: 113 = quad precision).

    Returns
    -------
    dict with keys:
        status        : "ok" or "unavailable" or "error"
        precision_bits: int
        R             : list[float]
        T             : list[float]
        wavelengths_nm: list[float]
    """
    if not _MPMATH_AVAILABLE:
        return {"status": "unavailable", "reason": "mpmath not installed"}

    N_lam = len(wavelengths_nm)

    # Resolve per-wavelength index table: n_table[medium_idx][lam_idx] -> complex
    if n_by_wavelength is not None:
        n_table: Sequence[Sequence[complex]] = n_by_wavelength
    elif n_all is not None:
        # Broadcast each scalar to all wavelengths (non-dispersive backward-compat path)
        n_table = [[complex(n)] * N_lam for n in n_all]
    else:
        return {"status": "error", "reason": "exactly one of n_all or n_by_wavelength must be provided"}

    theta0_rad = math.radians(angle_deg)
    pol = polarization.lower().strip()

    R_out: List[float] = []
    T_out: List[float] = []

    # Use a local workprec context so the global mpmath precision is not mutated.
    with _mp.workprec(precision_bits):
        mp = _mp
        theta0_mp = mp.mpf(theta0_rad)
        sin0 = mp.sin(theta0_mp)
        cos0 = mp.cos(theta0_mp)

        for i_lam, lam_nm in enumerate(wavelengths_nm):
            lam = mp.mpf(float(lam_nm))
            # Both lam and d_i are in nm — ratio is dimensionless, no conversion needed.

            n0 = mp.mpc(complex(n_table[0][i_lam]))
            n_exit = mp.mpc(complex(n_table[-1][i_lam]))
            film_n = [mp.mpc(complex(n_table[j + 1][i_lam])) for j in range(len(d_nm))]
            film_d = [mp.mpf(float(d)) for d in d_nm]

            # Incident medium admittance
            if pol == "s":
                eta0 = n0 * cos0
            else:
                eta0 = n0 / (cos0 + mp.mpf("1e-200"))

            # Exit medium admittance
            cos_exit = _cos_layer(n_exit, n0, sin0, mp)
            if pol == "s":
                eta_exit = n_exit * cos_exit
            else:
                eta_exit = n_exit / (cos_exit + mp.mpf("1e-200"))

            # Build characteristic matrix M = M_1 · M_2 · … · M_N
            M = mp.matrix([[mp.mpc(1), mp.mpc(0)], [mp.mpc(0), mp.mpc(1)]])

            for n_i, d_i in zip(film_n, film_d):
                cos_i = _cos_layer(n_i, n0, sin0, mp)
                if pol == "s":
                    eta_i = n_i * cos_i
                else:
                    eta_i = n_i / (cos_i + mp.mpf("1e-200"))

                delta = 2 * mp.pi * n_i * cos_i * d_i / lam
                c = mp.cos(delta)
                # -i*sin(delta): _cos_layer pins Im(n*cos) >= 0, so the Abeles
                # matrix must use the forward-decaying branch.  With +i an
                # absorbing layer amplifies, and this module is the acceptance
                # referee -- a sign error here silently certifies gain as loss.
                s = mp.mpc(0, -1) * mp.sin(delta)

                Mi = mp.matrix([
                    [c, s / (eta_i + mp.mpf("1e-200"))],
                    [s * eta_i, c],
                ])
                M = M * Mi

            # [B, C]^T = M · [1, eta_exit]^T
            B = M[0, 0] + M[0, 1] * eta_exit
            C = M[1, 0] + M[1, 1] * eta_exit

            denom = eta0 * B + C
            if abs(denom) < 1e-300:
                R_out.append(float("nan"))
                T_out.append(float("nan"))
                continue

            r = (eta0 * B - C) / denom
            t = 2 * eta0 / denom

            R_val = float(mp.re(abs(r) ** 2))
            T_val = float(mp.re(eta_exit) / mp.re(eta0)) * float(abs(t) ** 2)

            R_out.append(max(0.0, R_val))
            T_out.append(max(0.0, T_val))

    return {
        "status": "ok",
        "precision_bits": precision_bits,
        "R": R_out,
        "T": T_out,
        "wavelengths_nm": list(float(w) for w in wavelengths_nm),
    }


def run_referee(
    *,
    n_all: Optional[Sequence[complex]] = None,
    n_by_wavelength: Optional[Sequence[Sequence[complex]]] = None,
    d_nm: Sequence[float],
    wavelengths_nm: Sequence[float],
    angle_deg: float,
    polarization: str,
    primary_R: Optional[Sequence[float]] = None,
    primary_T: Optional[Sequence[float]] = None,
    secondary_R: Optional[Sequence[float]] = None,
    secondary_T: Optional[Sequence[float]] = None,
    precision_bits: int = PRECISION_BITS,
) -> Dict[str, Any]:
    """Run the high-precision referee and compare against both float64 solvers.

    Returns a dict suitable for embedding in the physics-acceptance certificate
    under the key ``high_precision_referee``.

    Keys in the returned dict
    -------------------------
    status                     : "ok" | "unavailable" | "error"
    precision_bits             : int
    max_abs_diff_from_primary  : float | None
    max_abs_diff_from_secondary: float | None
    closer_solver              : "primary" | "secondary" | "tied" | None
    R                          : list[float]
    T                          : list[float]
    """
    if not _MPMATH_AVAILABLE:
        return {
            "status": "unavailable",
            "reason": "mpmath not installed; install with: pip install mpmath>=1.3",
            "precision_bits": None,
            "max_abs_diff_from_primary": None,
            "max_abs_diff_from_secondary": None,
            "closer_solver": None,
        }

    try:
        hp = compute_rt_single_channel(
            n_all=n_all,
            n_by_wavelength=n_by_wavelength,
            d_nm=d_nm,
            wavelengths_nm=wavelengths_nm,
            angle_deg=angle_deg,
            polarization=polarization,
            precision_bits=precision_bits,
        )
        if hp["status"] != "ok":
            return {**hp, "max_abs_diff_from_primary": None,
                    "max_abs_diff_from_secondary": None, "closer_solver": None}

        import numpy as np
        hp_R = np.asarray(hp["R"], dtype=np.float64)
        hp_T = np.asarray(hp["T"], dtype=np.float64)

        def _max_rt_diff(R_ref: Optional[Sequence[float]], T_ref: Optional[Sequence[float]]) -> Optional[float]:
            if R_ref is None or T_ref is None:
                return None
            dR = float(np.max(np.abs(np.asarray(R_ref, dtype=np.float64) - hp_R)))
            dT = float(np.max(np.abs(np.asarray(T_ref, dtype=np.float64) - hp_T)))
            return max(dR, dT)

        diff_primary = _max_rt_diff(primary_R, primary_T)
        diff_secondary = _max_rt_diff(secondary_R, secondary_T)

        if diff_primary is not None and diff_secondary is not None:
            if diff_primary < diff_secondary * 0.9:
                closer = "primary"
            elif diff_secondary < diff_primary * 0.9:
                closer = "secondary"
            else:
                closer = "tied"
        else:
            closer = None

        return {
            "status": "ok",
            "precision_bits": precision_bits,
            "max_abs_diff_from_primary": diff_primary,
            "max_abs_diff_from_secondary": diff_secondary,
            "closer_solver": closer,
            "R": hp["R"],
            "T": hp["T"],
        }

    except Exception as exc:
        return {
            "status": "error",
            "reason": str(exc),
            "precision_bits": precision_bits,
            "max_abs_diff_from_primary": None,
            "max_abs_diff_from_secondary": None,
            "closer_solver": None,
        }


__all__ = [
    "PRECISION_BITS",
    "TRIGGER_THRESHOLD",
    "is_available",
    "compute_rt_single_channel",
    "run_referee",
]
