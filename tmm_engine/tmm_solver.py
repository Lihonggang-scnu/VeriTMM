# optimization.tmm_solver.py
# -*- coding: utf-8 -*-
"""
Debug-focused multilayer optics solver.

Provides:
  - reflectance_spectrum(): stable Abeles recursion (R only)
  - rt_spectrum(): stable S-matrix cascade (R/T)
  - rt_spectrum_cm(): characteristic matrix (can be numerically unstable; useful for diagnosis)

Convention:
  Fields ~ exp(i k z - i omega t). With that convention, passive absorption corresponds to Im(n) > 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TMMConfig:
    wavelength_unit: str = "um"          # "um" or "m"
    # Scalar or wavelength-resolved array.  Supporting an array lets the v2
    # workbench represent dispersive incident media without a Python loop.
    n_incident: Any = 1.0 + 0j           # incident medium (default: air)
    treat_last_layer_as_substrate: bool = True
    warn_if_out_of_bounds: bool = True
    out_of_bounds_tol: float = 1e-6
    clip_R: bool = True                 # keep historical behavior for reflectance_spectrum
    clip_T: bool = False                # debug: do NOT clip T by default


class TMM:
    def __init__(self, config: Optional[TMMConfig] = None):
        self.cfg = config or TMMConfig()
        if self.cfg.wavelength_unit not in ("um", "m"):
            raise ValueError("wavelength_unit must be 'um' or 'm'")

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _cos_theta(n_i: np.ndarray, n0: complex, sin_theta0: float) -> np.ndarray:
        """cos(theta_i) with a branch choice so that Im(kz) >= 0 for kz ~ n_i*cos(theta_i)."""
        sin_t = (n0 * sin_theta0) / (n_i + 1e-30)
        cos_t = np.sqrt(1.0 - sin_t * sin_t + 0j)
        kz = n_i * cos_t
        return np.where(np.imag(kz) < 0, -cos_t, cos_t)

    def _incident_n(self, count: int) -> np.ndarray:
        value = np.asarray(self.cfg.n_incident, dtype=np.complex128)
        if value.ndim == 0:
            return np.full(count, complex(value), dtype=np.complex128)
        value = value.reshape(-1)
        if value.shape != (count,):
            raise ValueError(
                f"wavelength-resolved n_incident must have shape {(count,)}, got {value.shape}"
            )
        return value

    @staticmethod
    def _eta(n_i: np.ndarray, cos_i: np.ndarray, pol: str) -> np.ndarray:
        """Admittance-like quantity for Fresnel coefficients."""
        if pol == "s":
            return n_i * cos_i
        if pol == "p":
            return n_i / (cos_i + 1e-30)
        raise ValueError("pol must be 's' or 'p'")

    @staticmethod
    def _interface_rt(eta_i: np.ndarray, eta_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interface r/t for E-field amplitude."""
        denom = eta_i + eta_j + 1e-30
        r_ij = (eta_i - eta_j) / denom
        t_ij = (2.0 * eta_i) / denom
        r_ji = -r_ij
        t_ji = (2.0 * eta_j) / denom
        return r_ij, t_ij, r_ji, t_ji

    @staticmethod
    def _star_product(
        SA11: np.ndarray, SA12: np.ndarray, SA21: np.ndarray, SA22: np.ndarray,
        SB11: np.ndarray, SB12: np.ndarray, SB21: np.ndarray, SB22: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Redheffer star product for cascading two 2-port scattering matrices A then B."""
        denom = (1.0 - SA22 * SB11 + 1e-30)
        S11 = SA11 + (SA12 * SB11 * SA21) / denom
        S12 = (SA12 * SB12) / denom
        S21 = (SB21 * SA21) / denom
        S22 = SB22 + (SB21 * SA22 * SB12) / denom
        return S11, S12, S21, S22

    def _warn_bounds(self, name: str, x: np.ndarray):
        if not self.cfg.warn_if_out_of_bounds:
            return
        tol = float(self.cfg.out_of_bounds_tol)
        x = np.asarray(x, dtype=float)
        too_high = np.mean(x > 1.0 + tol)
        too_low = np.mean(x < 0.0 - tol)
        if (too_high > 0.001) or (too_low > 0.001):
            print(
                f"[TMM warn] {name}: "
                f">{1.0+tol}: {too_high*100:.2f}%  "
                f"<{-tol}: {too_low*100:.2f}%"
            )

    # ---------------------------
    # Stable R via recursion
    # ---------------------------
    def reflectance_spectrum(
        self,
        n_list: List[np.ndarray],
        d_list: List[float],
        lam: np.ndarray,
        theta: float,
        pol: str,
        theta_unit: str = "rad",
        return_raw: bool = False,
    ) -> np.ndarray:
        pol = pol.lower().strip()
        if pol not in ("s", "p"):
            raise ValueError("pol must be 's' or 'p'")
        theta_unit = theta_unit.lower().strip()
        if theta_unit not in ("rad", "deg"):
            raise ValueError("theta_unit must be 'rad' or 'deg'")

        lam = np.asarray(lam, dtype=np.float64).ravel()
        N = lam.size
        if len(n_list) != len(d_list):
            raise ValueError("n_list and d_list must have same length")
        if len(n_list) < 1:
            raise ValueError("n_list must contain at least one layer (substrate)")

        lam_m = lam * 1e-6 if self.cfg.wavelength_unit == "um" else lam

        nk_layers = [np.asarray(nk, dtype=np.complex128).reshape(-1) for nk in n_list]
        for i, nk in enumerate(nk_layers):
            if nk.shape != (N,):
                raise ValueError(f"n_list[{i}] must have shape {(N,)}, got {nk.shape}")

        if self.cfg.treat_last_layer_as_substrate:
            nk_sub = nk_layers[-1]
            film_nk = nk_layers[:-1]
            film_d = [float(d) for d in d_list[:-1]]
        else:
            nk_sub = nk_layers[-1]
            film_nk = nk_layers
            film_d = [float(d) for d in d_list]

        n0 = self._incident_n(N)
        theta0 = float(theta)
        if theta_unit == "deg":
            theta0 = np.deg2rad(theta0)
        sin0 = float(np.sin(theta0))

        cos0 = np.cos(theta0) + 0j
        eta0 = (n0 * cos0) if pol == "s" else (n0 / (cos0 + 1e-30))

        cosS = self._cos_theta(nk_sub, n0, sin0)
        etaS = self._eta(nk_sub, cosS, pol)

        if len(film_nk) == 0:
            r = (eta0 - etaS) / (eta0 + etaS + 1e-30)
            R_raw = np.real(np.abs(r) ** 2)
            self._warn_bounds("R(raw)", R_raw)
            if return_raw or (not self.cfg.clip_R):
                return R_raw.astype(np.float64)
            return np.clip(R_raw, 0.0, 1.0).astype(np.float64)

        film_cos, film_eta = [], []
        for nk in film_nk:
            cos_i = self._cos_theta(nk, n0, sin0)
            film_cos.append(cos_i)
            film_eta.append(self._eta(nk, cos_i, pol))

        # effective reflection seen from last film into substrate
        r_eff = (film_eta[-1] - etaS) / (film_eta[-1] + etaS + 1e-30)

        # walk upward
        L = len(film_nk)
        for i in range(L - 2, -1, -1):
            nk_next = film_nk[i + 1]
            cos_next = film_cos[i + 1]
            d_next = film_d[i + 1]
            delta = (2.0 * np.pi) * nk_next * cos_next * (d_next / (lam_m + 1e-30))
            phase = np.exp(2j * delta)

            r_i = (film_eta[i] - film_eta[i + 1]) / (film_eta[i] + film_eta[i + 1] + 1e-30)
            r_eff = (r_i + r_eff * phase) / (1.0 + r_i * r_eff * phase + 1e-30)

        # include first layer propagation and incident interface
        nk1 = film_nk[0]
        cos1 = film_cos[0]
        d1 = film_d[0]
        delta1 = (2.0 * np.pi) * nk1 * cos1 * (d1 / (lam_m + 1e-30))
        phase1 = np.exp(2j * delta1)

        r01 = (eta0 - film_eta[0]) / (eta0 + film_eta[0] + 1e-30)
        r = (r01 + r_eff * phase1) / (1.0 + r01 * r_eff * phase1 + 1e-30)

        R_raw = np.real(np.abs(r) ** 2)
        self._warn_bounds("R(raw)", R_raw)
        if return_raw or (not self.cfg.clip_R):
            return R_raw.astype(np.float64)
        return np.clip(R_raw, 0.0, 1.0).astype(np.float64)

    # ---------------------------
    # Stable R/T via S-matrix
    # ---------------------------
    def rt_spectrum(
        self,
        n_list: List[np.ndarray],
        d_list: List[float],
        lam: np.ndarray,
        theta: float,
        pol: str,
        theta_unit: str = "rad",
        return_amplitudes: bool = False,
    ):
        pol = pol.lower().strip()
        if pol not in ("s", "p"):
            raise ValueError("pol must be 's' or 'p'")
        theta_unit = theta_unit.lower().strip()
        if theta_unit not in ("rad", "deg"):
            raise ValueError("theta_unit must be 'rad' or 'deg'")

        lam = np.asarray(lam, dtype=np.float64).ravel()
        N = lam.size
        if len(n_list) != len(d_list):
            raise ValueError("n_list and d_list must have same length")
        if len(n_list) < 1:
            raise ValueError("n_list must contain at least one layer (substrate)")
        lam_m = lam * 1e-6 if self.cfg.wavelength_unit == "um" else lam

        nk_layers = [np.asarray(nk, dtype=np.complex128).reshape(-1) for nk in n_list]
        for i, nk in enumerate(nk_layers):
            if nk.shape != (N,):
                raise ValueError(f"n_list[{i}] must have shape {(N,)}, got {nk.shape}")

        # Ports: incident medium -> film -> substrate (semi-infinite)
        if self.cfg.treat_last_layer_as_substrate:
            nk_sub = nk_layers[-1]
            film_nk = nk_layers[:-1]
            film_d = [float(d) for d in d_list[:-1]]
        else:
            nk_sub = nk_layers[-1]
            film_nk = nk_layers[:-1]
            film_d = [float(d) for d in d_list[:-1]]

        n0 = self._incident_n(N)
        theta0 = float(theta)
        if theta_unit == "deg":
            theta0 = np.deg2rad(theta0)
        sin0 = float(np.sin(theta0))

        cos0 = np.cos(theta0) + 0j
        eta0 = (n0 * cos0) if pol == "s" else (n0 / (cos0 + 1e-30))

        cosS = self._cos_theta(nk_sub, n0, sin0)
        etaS = self._eta(nk_sub, cosS, pol)

        film_cos, film_eta = [], []
        for nk in film_nk:
            cos_i = self._cos_theta(nk, n0, sin0)
            film_cos.append(cos_i)
            film_eta.append(self._eta(nk, cos_i, pol))

        etas = [np.asarray(eta0, dtype=np.complex128)] + film_eta + [etaS]

        # identity 2-port
        S11 = np.zeros(N, dtype=np.complex128)
        S22 = np.zeros(N, dtype=np.complex128)
        S12 = np.ones(N, dtype=np.complex128)
        S21 = np.ones(N, dtype=np.complex128)

        n_film = len(film_nk)
        for i in range(n_film + 1):
            eta_i = etas[i]
            eta_j = etas[i + 1]
            r_ij, t_ij, r_ji, t_ji = self._interface_rt(eta_i, eta_j)

            # interface scattering matrix
            I11, I12, I21, I22 = r_ij, t_ji, t_ij, r_ji
            S11, S12, S21, S22 = self._star_product(S11, S12, S21, S22, I11, I12, I21, I22)

            # propagation in the next layer if it's a film layer
            if i < n_film:
                nk = film_nk[i]
                cos_i = film_cos[i]
                d_i = film_d[i]
                delta = (2.0 * np.pi) * nk * cos_i * (d_i / (lam_m + 1e-30))
                p = np.exp(1j * delta)

                P11 = np.zeros(N, dtype=np.complex128)
                P22 = np.zeros(N, dtype=np.complex128)
                P12 = p
                P21 = p
                S11, S12, S21, S22 = self._star_product(S11, S12, S21, S22, P11, P12, P21, P22)

        r = S11
        t = S21

        R_raw = np.real(np.abs(r) ** 2)
        T_raw = (np.real(etaS) / (np.real(eta0) + 1e-30)) * (np.abs(t) ** 2)

        self._warn_bounds("R(raw)", R_raw)
        self._warn_bounds("T(raw)", T_raw)
        self._warn_bounds("R+T(raw)", R_raw + T_raw)

        R = np.clip(R_raw, 0.0, 1.0) if self.cfg.clip_R else R_raw
        T = np.clip(T_raw, 0.0, 1.0) if self.cfg.clip_T else T_raw

        if return_amplitudes:
            return R.astype(np.float64), T.astype(np.float64), r.astype(np.complex128), t.astype(np.complex128)
        return R.astype(np.float64), T.astype(np.float64)

    # ---------------------------
    # Characteristic matrix (diagnostic)
    # ---------------------------
    def rt_spectrum_cm(
        self,
        n_list: List[np.ndarray],
        d_list: List[float],
        lam: np.ndarray,
        theta: float,
        pol: str,
        theta_unit: str = "rad",
        return_amplitudes: bool = False,
    ):
        """
        Classic characteristic-matrix solver. This is mathematically fine,
        but can become numerically unstable for thick/absorbing stacks or deep-UV.
        Kept to compare against the stable S-matrix.

        Returns (R, T) or (R, T, r, t).
        """
        pol = pol.lower().strip()
        if pol not in ("s", "p"):
            raise ValueError("pol must be 's' or 'p'")
        theta_unit = theta_unit.lower().strip()
        if theta_unit not in ("rad", "deg"):
            raise ValueError("theta_unit must be 'rad' or 'deg'")

        lam = np.asarray(lam, dtype=np.float64).ravel()
        N = lam.size
        if len(n_list) != len(d_list):
            raise ValueError("n_list and d_list must have same length")
        if len(n_list) < 1:
            raise ValueError("n_list must contain at least one layer (substrate)")

        lam_m = lam * 1e-6 if self.cfg.wavelength_unit == "um" else lam

        nk_layers = [np.asarray(nk, dtype=np.complex128).reshape(-1) for nk in n_list]
        for i, nk in enumerate(nk_layers):
            if nk.shape != (N,):
                raise ValueError(f"n_list[{i}] must have shape {(N,)}, got {nk.shape}")

        if self.cfg.treat_last_layer_as_substrate:
            n_sub = nk_layers[-1]
            film_n = nk_layers[:-1]
            film_d = [float(d) for d in d_list[:-1]]
        else:
            n_sub = nk_layers[-1]
            film_n = nk_layers
            film_d = [float(d) for d in d_list]

        n0 = self._incident_n(N)
        theta0 = float(theta)
        if theta_unit == "deg":
            theta0 = np.deg2rad(theta0)
        sin0 = float(np.sin(theta0))
        cos0 = np.cos(theta0) + 0j

        def cos_theta(n_i):
            sin_t = (n0 * sin0) / (n_i + 1e-30)
            cos_t = np.sqrt(1.0 - sin_t * sin_t + 0j)
            kz = n_i * cos_t
            return np.where(np.imag(kz) < 0, -cos_t, cos_t)

        def eta(n_i, cos_i):
            if pol == "s":
                return n_i * cos_i
            return n_i / (cos_i + 1e-30)

        eta0 = (n0 * cos0) if pol == "s" else (n0 / (cos0 + 1e-30))
        cosS = cos_theta(n_sub)
        etaS = eta(n_sub, cosS)

        if len(film_n) == 0:
            r = (eta0 - etaS) / (eta0 + etaS + 1e-30)
            t = (2.0 * eta0) / (eta0 + etaS + 1e-30)
            R_raw = np.real(np.abs(r) ** 2)
            T_raw = (np.real(etaS) / (np.real(eta0) + 1e-30)) * (np.abs(t) ** 2)
            if return_amplitudes:
                return R_raw.astype(np.float64), T_raw.astype(np.float64), r.astype(np.complex128), t.astype(np.complex128)
            return R_raw.astype(np.float64), T_raw.astype(np.float64)

        # M = product of layer matrices
        M11 = np.ones(N, dtype=np.complex128)
        M12 = np.zeros(N, dtype=np.complex128)
        M21 = np.zeros(N, dtype=np.complex128)
        M22 = np.ones(N, dtype=np.complex128)

        for n_i, d_i in zip(film_n, film_d):
            cos_i = cos_theta(n_i)
            eta_i = eta(n_i, cos_i)
            delta = (2.0 * np.pi) * n_i * cos_i * (d_i / (lam_m + 1e-30))

            # Sign convention: cos_theta above pins Im(n*cos) >= 0, the same
            # forward-decaying branch the stable path uses via exp(+1j*delta).
            # The Abeles matrix must sit on that branch too, so the off-diagonal
            # term is -1j*sin(delta); with +1j an absorbing layer amplifies
            # instead of attenuating and R+T exceeds 1 without bound.
            # Still overflows for large |Im(delta)|; kept for diagnosis.
            c = np.cos(delta)
            s = -1j * np.sin(delta)

            a11 = c
            a12 = s / (eta_i + 1e-30)
            a21 = s * eta_i
            a22 = c

            t11 = M11 * a11 + M12 * a21
            t12 = M11 * a12 + M12 * a22
            t21 = M21 * a11 + M22 * a21
            t22 = M21 * a12 + M22 * a22
            M11, M12, M21, M22 = t11, t12, t21, t22

        denom = (eta0 * M11 + eta0 * etaS * M12 + M21 + etaS * M22)
        r = (eta0 * M11 + eta0 * etaS * M12 - M21 - etaS * M22) / (denom + 1e-30)
        t = (2.0 * eta0) / (denom + 1e-30)

        R_raw = np.real(np.abs(r) ** 2)
        T_raw = (np.real(etaS) / (np.real(eta0) + 1e-30)) * (np.abs(t) ** 2)

        self._warn_bounds("R_cm(raw)", R_raw)
        self._warn_bounds("T_cm(raw)", T_raw)
        self._warn_bounds("R+T_cm(raw)", R_raw + T_raw)

        R = np.clip(R_raw, 0.0, 1.0) if self.cfg.clip_R else R_raw
        T = np.clip(T_raw, 0.0, 1.0) if self.cfg.clip_T else T_raw

        if return_amplitudes:
            return R.astype(np.float64), T.astype(np.float64), r.astype(np.complex128), t.astype(np.complex128)
        return R.astype(np.float64), T.astype(np.float64)

    # ---------------------------------------------------------------------------------
    # Backward-compatible wrappers (match your original tmm_solver.py API)
    # ---------------------------------------------------------------------------------
    def _stack_layers_to_lists(
        self,
        stack_layers: List[Dict],
        get_nk: Callable[[str, np.ndarray], np.ndarray],
        lam: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Convert stack_layers (list of dicts) into (n_list, d_list).

        stack_layers element format (as in your original project):
            {"material": "tio2", "thickness_m": 88e-9}
        The last layer is treated as substrate if cfg.treat_last_layer_as_substrate=True.
        """
        n_list: List[np.ndarray] = []
        d_list: List[float] = []
        for L in stack_layers:
            mat = str(L.get("material", "")).lower().strip()
            d_m = float(L.get("thickness_m", 0.0))
            nk = np.asarray(get_nk(mat, lam), dtype=np.complex128).reshape(-1)
            n_list.append(nk)
            d_list.append(d_m)
        return n_list, d_list

    def reflectance_from_stack(
        self,
        stack_layers: List[Dict],
        get_nk: Callable[[str, np.ndarray], np.ndarray],
        lam: np.ndarray,
        theta: float,
        pol: str,
        theta_unit: str = "rad",
        return_raw: bool = False,
    ) -> np.ndarray:
        """Compute R(位) for a stack. return_raw is accepted for convenience.

        Note: if you want *raw* (unclipped) values, set cfg.clip_R=False.
        """
        n_list, d_list = self._stack_layers_to_lists(stack_layers, get_nk, lam)
        # reflectance_spectrum already applies cfg.clip_R; return_raw kept for API compat.
        return self.reflectance_spectrum(n_list, d_list, lam, theta=theta, pol=pol, theta_unit=theta_unit)

    def rt_from_stack(
        self,
        stack_layers: List[Dict],
        get_nk: Callable[[str, np.ndarray], np.ndarray],
        lam: np.ndarray,
        theta: float,
        pol: str,
        theta_unit: str = "rad",
        return_amplitudes: bool = False,
    ):
        """Compute (R,T) using the numerically-stable scattering-matrix (recommended)."""
        n_list, d_list = self._stack_layers_to_lists(stack_layers, get_nk, lam)
        return self.rt_spectrum(
            n_list,
            d_list,
            lam,
            theta=theta,
            pol=pol,
            theta_unit=theta_unit,
            return_amplitudes=return_amplitudes,
        )
